import pickle, csv, torch, time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util

# === Config ===
llm_model_id = "meta-llama/Llama-3.1-8B-Instruct"
nli_model_id = "roberta-large-mnli"
output_csv = "ablation.csv"
stats_file = "model_stats_ablation.csv"
debug = True
entail_threshold = 0.5
fallback_top_k = 3
run_mode = "hybrid"  # Options: "abstractive", "extractive", "hybrid"

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
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Load Dataset ===
print("Loading dataset...")
with open("data.pkl", "rb") as f:
    dataset = pickle.load(f)

few_shot_examples = dataset[:3]
eval_dataset = dataset[3:]

# === Helper Functions ===
def entails(premise, hypothesis, threshold=entail_threshold):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = softmax(logits, dim=1)
    return probs[0][2].item() > threshold

def normalize(text):
    return text.lower().strip()

# === Inference ===
print("Starting inference...")
start_time = time.time()
torch.cuda.reset_peak_memory_stats()

csv_rows, predictions, references = [], [], []
smoothie = SmoothingFunction().method4

for i, item in enumerate(eval_dataset):
    qid = item["question_id"]
    title = item["question_title"]
    answer = item["answer_body"]
    sentences = item["sentences"]

    messages = [{"role": "system", "content": "You are an expert summarizer. Generate concise summaries for Stack Overflow answers."}]
    for ex in few_shot_examples:
        q = ex["question_title"]
        a = ex["answer_body"]
        gt = " ".join([s["sentence"] for s in ex["sentences"] if s.get("truth") == 1])
        messages.append({"role": "user", "content": f"Question: {q}\nAnswer: {a}\nSummary:"})
        messages.append({"role": "assistant", "content": gt.strip()})
    messages.append({"role": "user", "content": f"Question: {title}\nAnswer: {answer}\nSummary:"})
    input_text = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    if debug:
        print(f"\nProcessing item {i+1}/{len(eval_dataset)} (ID: {qid})")
        print("="*50)
        print("Input prompt:")
        print(input_text[:500] + "..." if len(input_text) > 500 else input_text)
        print("="*50)

    try:
        outputs = llm.generate(input_text, sampling_params)
        abstractive = outputs[0].outputs[0].text.strip()
        if debug:
            print(f"Generated abstractive summary:\n{abstractive}")
    except Exception as e:
        print(f"Error generating for {qid}: {str(e)}")
        abstractive = ""

    extracted = [s["sentence"] for s in sentences if entails(s["sentence"], abstractive)]

    if not extracted and fallback_top_k:
        sent_texts = [s["sentence"] for s in sentences]
        abstr_emb = embedder.encode(abstractive, convert_to_tensor=True)
        sent_embs = embedder.encode(sent_texts, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(abstr_emb.to("cuda"), sent_embs.to("cuda"))[0]
        #cosine_scores = util.pytorch_cos_sim(abstr_emb, sent_embs)[0]
        top_indices = torch.topk(cosine_scores, k=min(fallback_top_k, len(sentences))).indices.tolist()
        extracted = [sent_texts[i] for i in top_indices]

    extractive = " ".join(extracted)
    ground_truth = " ".join([s["sentence"] for s in sentences if s.get("truth") == 1])

    if debug:
        print(f"\nExtractive summary:\n{extractive}")
        print(f"\nGround truth:\n{ground_truth}")
        print("="*50)

    if run_mode == "abstractive":
        predictions.append(abstractive)
    elif run_mode == "extractive":
        predictions.append(extractive)
    else:
        predictions.append(extractive)

    references.append(ground_truth)
    csv_rows.append([qid, title, answer, abstractive, extractive, ground_truth])

# === Save Output ===
print("Saving results...")
with open(output_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Question ID", "Title", "Answer Body", "Abstractive Summary", "Extractive Summary", "Ground Truth"])
    writer.writerows(csv_rows)

# === Evaluation ===
print("Calculating metrics...")
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

for ref, pred in zip(references, predictions):
    if not pred.strip(): continue
    scores = rouge.score(ref, pred)
    rouge1_scores.append(scores["rouge1"].fmeasure)
    rouge2_scores.append(scores["rouge2"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

predictions_norm = [normalize(p) for p in predictions]
references_norm = [normalize(r) for r in references]
P, R, F1 = bert_score(predictions_norm, references_norm, lang="en")

bleu_scores = [
    sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
    for ref, pred in zip(references, predictions) if pred.strip()
]

elapsed = time.time() - start_time
mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
mem_reserved = torch.cuda.max_memory_reserved() / 1024**3

rouge1_avg = sum(rouge1_scores)/len(rouge1_scores) if rouge1_scores else 0.0
rouge2_avg = sum(rouge2_scores)/len(rouge2_scores) if rouge2_scores else 0.0
rougeL_avg = sum(rougeL_scores)/len(rougeL_scores) if rougeL_scores else 0.0
bleu_avg = sum(bleu_scores)/len(bleu_scores) if bleu_scores else 0.0

print("\n=== Final Results ===")
print(f"?? Time: {elapsed:.2f} sec | \U0001f9e0 Mem Alloc: {mem_alloc:.2f} GB | Reserved: {mem_reserved:.2f} GB")
print(f"\U0001f4ca ROUGE-1 F1: {rouge1_avg:.4f}")
print(f"\U0001f4ca ROUGE-2 F1: {rouge2_avg:.4f}")
print(f"\U0001f4ca ROUGE-L F1: {rougeL_avg:.4f}")
print("\U0001f916 BERTScore:")
print(f"  Precision: {P.mean().item():.4f}")
print(f"  Recall:    {R.mean().item():.4f}")
print(f"  F1:        {F1.mean().item():.4f}")
print(f"?? BLEU: {bleu_avg:.4f}")

with open(stats_file, "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        llm_model_id, f"{elapsed:.2f}", f"{mem_alloc:.2f}", f"{mem_reserved:.2f}",
        f"{rouge1_avg:.4f}", f"{rouge2_avg:.4f}", f"{rougeL_avg:.4f}",
        f"{F1.mean().item():.4f}", f"{bleu_avg:.4f}"
    ])

print("Done!")
