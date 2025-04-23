import pickle, csv, torch, time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# === Config ===
llm_model_id = "meta-llama/Llama-3.1-8B-Instruct"
nli_model_id = "roberta-large-mnli"
output_csv = "llama8b_fp8_one_shot.csv"
stats_file = "model_stats.csv"
debug = True  # Set to True to print debug information

# === Load Models ===
print("Loading models...")
llm = LLM(
    model=llm_model_id,
    tensor_parallel_size=2,
    quantization="fp8",
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    swap_space=16
)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_id).half().to("cuda")

# === Load Dataset ===
print("Loading dataset...")
with open("data.pkl", "rb") as f:
    dataset = pickle.load(f)

# Split into few-shot examples and evaluation set
few_shot_examples = dataset[:1]  # Use first 1 item as few-shot examples
eval_dataset = dataset[1:]       # Evaluate on the rest

# === Few-shot Prompt Construction ===
print("Constructing one-shot prompt...")
few_shot_prompt = """You are an expert at summarizing answers to technical questions. 
Given a question and its answer, create a concise summary that captures the key information.

Here are some examples:\n\n"""

for ex in few_shot_examples:
    q = ex["question_title"]
    a = ex["answer_body"]
    gt = " ".join([s["sentence"] for s in ex["sentences"] if s.get("truth") == 1])
    few_shot_prompt += f"""### Example
Question: {q}
Answer: {a}
Summary: {gt.strip()}\n\n"""

few_shot_prompt += """Now create a summary for this new example:
### Example
"""

# === Entailment Function ===
def entails(premise, hypothesis, threshold=0.6):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = softmax(logits, dim=1)
    return probs[0][2].item() > threshold  # index 2 is entailment

# === Run Inference ===
print("Starting inference...")
start_time = time.time()
torch.cuda.reset_peak_memory_stats()

csv_rows = []
predictions, references = [], []
smoothie = SmoothingFunction().method4

for i, item in enumerate(eval_dataset):
    qid = item["question_id"]
    title = item["question_title"]
    answer = item["answer_body"]
    sentences = item["sentences"]

    # Construct prompt
    prompt = few_shot_prompt + f"Question: {title}\nAnswer: {answer}\nSummary:"
    messages = [{"role": "user", "content": prompt}]
    input_text = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    if debug:
        print(f"\nProcessing item {i+1}/{len(eval_dataset)} (ID: {qid})")
        print("="*50)
        print("Input prompt:")
        print(input_text[:500] + "..." if len(input_text) > 500 else input_text)
        print("="*50)

    # Generate abstractive summary
    try:
        outputs = llm.generate(input_text, sampling_params)
        abstractive = outputs[0].outputs[0].text.strip()
        if debug:
            print(f"Generated abstractive summary:\n{abstractive}")
    except Exception as e:
        print(f"Error generating for {qid}: {str(e)}")
        abstractive = ""

    # Create extractive summary using entailment
    extractive = " ".join(
        s["sentence"] for s in sentences if entails(abstractive, s["sentence"])  # Fixed entailment direction
    )
    
    # Get ground truth
    ground_truth = " ".join([s["sentence"] for s in sentences if s.get("truth") == 1])
    
    if debug:
        print(f"\nExtractive summary:\n{extractive}")
        print(f"\nGround truth:\n{ground_truth}")
        print("="*50)

    # Store results
    csv_rows.append([qid, title, answer, abstractive, extractive, ground_truth])
    predictions.append(extractive)
    references.append(ground_truth)

# === Save Output ===
print("Saving results...")
with open(output_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Question ID", "Title", "Answer Body", "Abstractive Summary", "Extractive Summary", "Ground Truth"])
    writer.writerows(csv_rows)

# === Evaluation ===
print("Calculating metrics...")
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for ref, pred in zip(references, predictions):
    if not pred.strip():  # Skip empty predictions
        continue
    scores = rouge.score(ref, pred)
    rouge1_scores.append(scores["rouge1"].fmeasure)
    rouge2_scores.append(scores["rouge2"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

# Calculate BERTScore
P, R, F1 = bert_score(predictions, references, lang="en")

# Calculate BLEU scores
bleu_scores = []
for ref, pred in zip(references, predictions):
    if not pred.strip():  # Skip empty predictions
        continue
    ref_tokens = ref.split()
    pred_tokens = pred.split()
    bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie))

# === Stats ===
elapsed = time.time() - start_time
mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
mem_reserved = torch.cuda.max_memory_reserved() / 1024**3

# Handle cases where all predictions were empty
if not rouge1_scores:
    rouge1_avg = rouge2_avg = rougeL_avg = 0.0
else:
    rouge1_avg = sum(rouge1_scores)/len(rouge1_scores)
    rouge2_avg = sum(rouge2_scores)/len(rouge2_scores)
    rougeL_avg = sum(rougeL_scores)/len(rougeL_scores)

if not bleu_scores:
    bleu_avg = 0.0
else:
    bleu_avg = sum(bleu_scores)/len(bleu_scores)

# Print results
print("\n=== Final Results ===")
print(f"?? Time: {elapsed:.2f} sec | ?? Mem Alloc: {mem_alloc:.2f} GB | Reserved: {mem_reserved:.2f} GB")
print(f"?? ROUGE-1 F1: {rouge1_avg:.4f}")
print(f"?? ROUGE-2 F1: {rouge2_avg:.4f}")
print(f"?? ROUGE-L F1: {rougeL_avg:.4f}")
print(f"?? BERTScore:")
print(f"  Precision: {P.mean().item():.4f}")
print(f"  Recall:    {R.mean().item():.4f}")
print(f"  F1:        {F1.mean().item():.4f}")
print(f"?? BLEU: {bleu_avg:.4f}")

# === Save Stats ===
with open(stats_file, "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        llm_model_id,
        f"{elapsed:.2f}", f"{mem_alloc:.2f}", f"{mem_reserved:.2f}",
        f"{rouge1_avg:.4f}",
        f"{rouge2_avg:.4f}",
        f"{rougeL_avg:.4f}",
        f"{F1.mean().item():.4f}",
        f"{bleu_avg:.4f}"
    ])

print("Done!")