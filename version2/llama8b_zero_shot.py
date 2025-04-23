import pickle
import csv
import torch
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datetime import datetime

# === Config ===
ablation_mode = "hybrid"  # Options: "abstractive", "extractive", "hybrid"
top_k_sentences = 2
llm_model_id = "meta-llama/Llama-3.1-8B-Instruct"
nli_model_id = "roberta-large-mnli"

# Output filenames
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = f"llama8b_zero_shot_{ablation_mode}_{run_id}.csv"
stats_file = f"model_stats_zero_shot_{run_id}.csv"

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

# === Entailment + Top-K fallback ===
def entails(premise, hypothesis, threshold=0.6):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = softmax(logits, dim=1)
    return probs[0][2].item() > threshold, probs[0][2].item()

# === Run Inference ===
print("Starting inference...")
start_time = time.time()
torch.cuda.reset_peak_memory_stats()

csv_rows = []
predictions, references = [], []
smoothie = SmoothingFunction().method4

for item in dataset:
    qid = item["question_id"]
    title = item["question_title"]
    answer = item["answer_body"]
    sentences = item["sentences"]

    # --- Abstractive summary ---
    abstractive = ""
    if ablation_mode in ["abstractive", "hybrid"]:
        messages = [
            {"role": "system", "content": "You are an expert summarizer. Generate a concise summary."},
            {"role": "user", "content": f"Question: {title}\nAnswer: {answer}\nSummary:"}
        ]
        prompt = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        try:
            outputs = llm.generate(prompt, sampling_params)
            abstractive = outputs[0].outputs[0].text.strip()
        except Exception as e:
            abstractive = "ERROR: " + str(e)

    # --- Extractive summary ---
    extractive = ""
    if ablation_mode in ["extractive", "hybrid"] and abstractive:
        entailed = []
        sentence_scores = []
        for s in sentences:
            flag, score = entails(s["sentence"], abstractive)
            if flag:
                entailed.append(s["sentence"])
            sentence_scores.append((score, s["sentence"]))

        if not entailed:
            sentence_scores.sort(reverse=True)
            entailed = [s for _, s in sentence_scores[:top_k_sentences]]

        extractive = " ".join(entailed)

    # --- Ground Truth ---
    ground_truth = " ".join([s["sentence"] for s in sentences if s.get("truth") == 1])

    predictions.append(extractive if ablation_mode != "abstractive" else abstractive)
    references.append(ground_truth)

    csv_rows.append([qid, title, answer, abstractive, extractive, ground_truth])

# === Save CSV ===
with open(output_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Question ID", "Title", "Answer Body", "Abstractive Summary", "Extractive Summary", "Ground Truth"])
    writer.writerows(csv_rows)

# === Metrics ===
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1, rouge2, rougel = [], [], []
for ref, pred in zip(references, predictions):
    scores = rouge.score(ref, pred)
    rouge1.append(scores["rouge1"].fmeasure)
    rouge2.append(scores["rouge2"].fmeasure)
    rougel.append(scores["rougeL"].fmeasure)

P, R, F1 = bert_score(predictions, references, lang="en")
bleu = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie) for ref, pred in zip(references, predictions)]

# === Stats ===
elapsed = time.time() - start_time
mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
mem_reserved = torch.cuda.max_memory_reserved() / 1024**3

print("\n=== Final Results ===")
print(f"?? Time: {elapsed:.2f} sec | ?? Mem Alloc: {mem_alloc:.2f} GB | Reserved: {mem_reserved:.2f} GB")
print(f"?? ROUGE-1 F1: {sum(rouge1)/len(rouge1):.4f}")
print(f"?? ROUGE-2 F1: {sum(rouge2)/len(rouge2):.4f}")
print(f"?? ROUGE-L F1: {sum(rougel)/len(rougel):.4f}")
print("?? BERTScore:")
print(f"  Precision: {P.mean():.4f}")
print(f"  Recall:    {R.mean():.4f}")
print(f"  F1:        {F1.mean():.4f}")
print(f"?? BLEU: {sum(bleu)/len(bleu):.4f}")

with open(stats_file, "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        llm_model_id,
        f"{elapsed:.2f}", f"{mem_alloc:.2f}", f"{mem_reserved:.2f}",
        f"{sum(rouge1)/len(rouge1):.4f}", f"{sum(rouge2)/len(rouge2):.4f}", f"{sum(rougel)/len(rougel):.4f}",
        f"{F1.mean():.4f}", f"{sum(bleu)/len(bleu):.4f}"
    ])

print("Done!")
