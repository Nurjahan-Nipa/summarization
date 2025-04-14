import pickle
import csv
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# ========== 1. Load LLaMA 3.3 70B and Tokenizer ==========
llm_model_id = "meta-llama/Llama-3.3-70B-Instruct"
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.95,
    max_model_len=2048  
)


#llm = LLM(model=llm_model_id, tensor_parallel_size=2)  # Adjust based on GPU availability
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

# ========== 2. Load NLI Model for Entailment ==========
nli_model_id = "roberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_id)

# ========== 3. Load Dataset ==========
with open("data.pkl", "rb") as f:
    dataset = pickle.load(f)

# ========== 4. Entailment Function ==========
def entails(premise, hypothesis, threshold=0.6):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = softmax(logits, dim=1)
    entail_prob = probs[0][2].item()  # label 2 = entailment
    return entail_prob > threshold

# ========== 5. Summarization + NLI Filtering ==========
csv_rows = []
predictions, references = [], []

for item in dataset:
    qid = item.get("question_id")
    title = item.get("question_title", "")
    answer = item.get("answer_body", "")
    sentences = item.get("sentences", [])

    # --- Abstractive Summary with LLaMA ---
    messages = [
        {"role": "system", "content": "You are an expert summarizer. Summarize Stack Overflow answers concisely."},
        {"role": "user", "content": f"Question: {title}\n\nAnswer: {answer}\n\nSummary:"}
    ]
    prompt = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    try:
        outputs = llm.generate(prompt, sampling_params)
        abstractive_summary = outputs[0].outputs[0].text.strip()
    except Exception as e:
        abstractive_summary = f"ERROR: {e}"

    # --- Extractive Summary via NLI ---
    extracted_sentences = []
    for sent_obj in sentences:
        sentence = sent_obj["sentence"]
        if entails(abstractive_summary, sentence):
            extracted_sentences.append(sentence)
    extractive_summary = " ".join(extracted_sentences)

    # --- Ground Truth Summary ---
    ground_truth = " ".join([s["sentence"] for s in sentences if s.get("truth") == 1])

    # --- Collect for CSV and Evaluation ---
    csv_rows.append([qid, title, answer, abstractive_summary, extractive_summary, ground_truth])
    predictions.append(extractive_summary)
    references.append(ground_truth)

# ========== 6. Save Summaries to CSV ==========
with open("assortis_llama3_70b.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Question ID", "Title", "Answer Body", "Abstractive Summary", "Extractive Summary", "Ground Truth"])
    writer.writerows(csv_rows)

print("✅ Summarization completed. Saved to assortis_llama3_70b.csv")

# ========== 7. Evaluate with ROUGE ==========
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1, rouge2, rougel = [], [], []

for ref, pred in zip(references, predictions):
    scores = rouge.score(ref, pred)
    rouge1.append(scores["rouge1"].fmeasure)
    rouge2.append(scores["rouge2"].fmeasure)
    rougel.append(scores["rougeL"].fmeasure)

print("\n📊 ROUGE Scores:")
print(f"ROUGE-1 F1: {sum(rouge1)/len(rouge1):.4f}")
print(f"ROUGE-2 F1: {sum(rouge2)/len(rouge2):.4f}")
print(f"ROUGE-L F1: {sum(rougel)/len(rougel):.4f}")

# ========== 8. Evaluate with BERTScore ==========
P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)
print("\n📊 BERTScore:")
print(f"Precision: {P.mean():.4f}")
print(f"Recall:    {R.mean():.4f}")
print(f"F1:        {F1.mean():.4f}")

