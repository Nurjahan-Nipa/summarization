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

# ========== Configuration ==========
llm_model_id = "meta-llama/Llama-3.1-8B-Instruct"
nli_model_id = "roberta-large-mnli"
stats_file = "model_stats.csv"

# ========== 1. Load LLaMA 3.1 8B ==========
llm = LLM(model=llm_model_id, tensor_parallel_size=2,  max_model_len=2048, gpu_memory_utilization=0.85, max_num_seqs=4, enforce_eager=True, swap_space=16, quantization="fp8")
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

# ========== 2. Load RoBERTa NLI Model ==========
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_id).half().to("cuda")

# ========== 3. Load Dataset ==========
with open("data.pkl", "rb") as f:
    dataset = pickle.load(f)

# ========== 4. Entailment Function ==========
def entails(premise, hypothesis, threshold=0.6):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = softmax(logits, dim=1)
    entail_prob = probs[0][2].item()
    return entail_prob > threshold

# ========== 5. Inference Loop with Metrics ==========
csv_rows = []
predictions, references = [], []

start_time = time.time()
torch.cuda.reset_peak_memory_stats()

for item in dataset:
    qid = item.get("question_id")
    title = item.get("question_title", "")
    answer = item.get("answer_body", "")
    sentences = item.get("sentences", [])

    # Abstractive Summarization
    messages = [
        {"role": "system", "content": "Your task is to generate a brief and accurate summary of the provided Stack Overflow answer, based on the question context"},
        {"role": "user", "content": f"Question: {title}\n\nAnswer: {answer}\n\nSummary:"}
    ]
    prompt = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    try:
        outputs = llm.generate(prompt, sampling_params)
        abstractive_summary = outputs[0].outputs[0].text.strip()
    except Exception as e:
        abstractive_summary = f"ERROR: {e}"

    # Extractive Summary via NLI
    extracted_sentences = []
    for sent_obj in sentences:
        sentence = sent_obj["sentence"]
        if entails(abstractive_summary, sentence):
            extracted_sentences.append(sentence)
    extractive_summary = " ".join(extracted_sentences)

    # Ground Truth Summary
    ground_truth = " ".join([s["sentence"] for s in sentences if s.get("truth") == 1])

    # Save results
    csv_rows.append([qid, title, answer, abstractive_summary, extractive_summary, ground_truth])
    predictions.append(extractive_summary)
    references.append(ground_truth)

# ========== 6. Time and GPU Stats ==========
end_time = time.time()
elapsed_time = end_time - start_time
mem_allocated = torch.cuda.max_memory_allocated() / (1024**3)
mem_reserved = torch.cuda.max_memory_reserved() / (1024**3)

# ========== 7. Save Summaries ==========
with open("llama3_8b_base.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Question ID", "Title", "Answer Body", "Abstractive Summary", "Extractive Summary", "Ground Truth"])
    writer.writerows(csv_rows)

print("\n Summarization completed")

# ========== 8. Evaluate ROUGE ==========
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1, rouge2, rougel = [], [], []

for ref, pred in zip(references, predictions):
    scores = rouge.score(ref, pred)
    rouge1.append(scores["rouge1"].fmeasure)
    rouge2.append(scores["rouge2"].fmeasure)
    rougel.append(scores["rougeL"].fmeasure)

avg_rouge1 = sum(rouge1)/len(rouge1)
avg_rouge2 = sum(rouge2)/len(rouge2)
avg_rougel = sum(rougel)/len(rougel)

print("\n📊 ROUGE Scores:")
print(f"ROUGE-1 F1: {avg_rouge1:.4f}")
print(f"ROUGE-2 F1: {avg_rouge2:.4f}")
print(f"ROUGE-L F1: {avg_rougel:.4f}")

# ========== 9. Evaluate BERTScore ==========
P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)
bert_f1 = F1.mean().item()

print("\n📊 BERTScore:")
print(f"Precision: {P.mean():.4f}")
print(f"Recall:    {R.mean():.4f}")
print(f"F1:        {bert_f1:.4f}")

# ========== 10. Evaluate BLEU Score ==========
bleu_scores = []
smoothie = SmoothingFunction().method4  # Handles short outputs

for ref, pred in zip(references, predictions):
    bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
    bleu_scores.append(bleu)

avg_bleu = sum(bleu_scores) / len(bleu_scores)

print("\n?? BLEU Score:")
print(f"BLEU: {avg_bleu:.4f}")

# ========== 11. Save Evaluation Stats ==========
# ========== 11. Save Evaluation Stats ==========
with open(stats_file, "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        llm_model_id,
        f"{elapsed_time:.2f}",
        f"{mem_allocated:.2f}",
        f"{mem_reserved:.2f}",
        f"{avg_rouge1:.4f}",
        f"{avg_rouge2:.4f}",
        f"{avg_rougel:.4f}",
        f"{bert_f1:.4f}",
        f"{avg_bleu:.4f}"  # Add BLEU here
    ])


print("\n📥 Stats saved to model_stats.csv")

