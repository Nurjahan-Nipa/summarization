
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
output_csv = "llamav2.csv"
stats_file = "model_stats_llamv2.csv"

mode = "hybrid"       # Options: "abstractive", "extractive", "hybrid"
top_k = 3             # Top-k sentence selection; set to None for threshold-based
entail_threshold = 0.6

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

print("Loading dataset...")
with open("data.pkl", "rb") as f:
    dataset = pickle.load(f)

few_shot_prompt = """You are an expert summarizer. Generate concise summaries for Stack Overflow answers.\n\n"""
for ex in dataset[:3]:
    q = ex["question_title"]
    a = ex["answer_body"]
    gt = " ".join([s["sentence"] for s in ex["sentences"] if s.get("truth") == 1])
    few_shot_prompt += f"### Example\nQuestion: {q}\nAnswer: {a}\nSummary: {gt.strip()}\n\n"

smoothie = SmoothingFunction().method4

    
def get_entail_scores(premise, candidate_sents):
    if not candidate_sents:  # Prevent crash on empty input
        return []
    inputs = nli_tokenizer([premise] * len(candidate_sents), candidate_sents, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = softmax(logits, dim=1)
    entail_scores = probs[:, 2].tolist()
    return entail_scores
    
def select_extractive(summary, sentences):
    candidates = [s["sentence"] for s in sentences]
    if not candidates:
        return ""

    scores = get_entail_scores(summary, candidates)
    
    if not scores:
        return ""

    if top_k:
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return " ".join([candidates[i] for i in top_indices])
    else:
        return " ".join([s for s, score in zip(candidates, scores) if score > entail_threshold])

start = time.time()
torch.cuda.reset_peak_memory_stats()
csv_rows, predictions, references = [], [], []

for item in dataset[3:]:
    qid = item["question_id"]
    title = item["question_title"]
    answer = item["answer_body"]
    sentences = item["sentences"]

    abstractive = ""
    if mode in ["abstractive", "hybrid"]:
        messages = [{"role": "user", "content": few_shot_prompt + f"### Example\nQuestion: {title}\nAnswer: {answer}\nSummary:"}]
        prompt = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        try:
            outputs = llm.generate(prompt, sampling_params)
            abstractive = outputs[0].outputs[0].text.strip()
        except Exception as e:
            abstractive = f"ERROR: {str(e)}"

    extractive = ""
    if mode in ["extractive", "hybrid"] and abstractive:
        extractive = select_extractive(abstractive, sentences).strip()

    # Fallback if extractive is empty
    if mode == "hybrid" and not extractive and abstractive:
        extractive = abstractive

    ground_truth = " ".join([s["sentence"] for s in sentences if s.get("truth") == 1])
    csv_rows.append([qid, title, answer, abstractive, extractive, ground_truth])
    predictions.append(extractive)
    references.append(ground_truth)

# Save outputs
with open(output_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Question ID", "Title", "Answer Body", "Abstractive Summary", "Extractive Summary", "Ground Truth"])
    writer.writerows(csv_rows)

# Evaluation
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1, rouge2, rougeL = [], [], []
for ref, pred in zip(references, predictions):
    if pred.strip():
        scores = rouge.score(ref, pred)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)

P, R, F1 = bert_score(predictions, references, lang="en")
bleu = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie) for ref, pred in zip(references, predictions) if pred.strip()]

elapsed = time.time() - start
mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
mem_reserved = torch.cuda.max_memory_reserved() / 1024**3

print("\n=== Metrics ===")
print(f"ROUGE-1: {sum(rouge1)/len(rouge1):.4f}")
print(f"ROUGE-2: {sum(rouge2)/len(rouge2):.4f}")
print(f"ROUGE-L: {sum(rougeL)/len(rougeL):.4f}")
print(f"  Precision: {P.mean().item():.4f}")
print(f"  Recall:    {R.mean().item():.4f}")
print(f"  F1:        {F1.mean().item():.4f}")
print(f"BLEU: {sum(bleu)/len(bleu):.4f}")
print(f"?? Time: {elapsed:.2f}s | ?? Mem Alloc: {mem_alloc:.2f} GB | Reserved: {mem_reserved:.2f} GB")

with open(stats_file, "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        llm_model_id, f"{elapsed:.2f}", f"{mem_alloc:.2f}", f"{mem_reserved:.2f}",
        f"{sum(rouge1)/len(rouge1):.4f}", f"{sum(rouge2)/len(rouge2):.4f}", f"{sum(rougeL)/len(rougeL):.4f}",
        f"{F1.mean():.4f}", f"{sum(bleu)/len(bleu):.4f}"
    ])