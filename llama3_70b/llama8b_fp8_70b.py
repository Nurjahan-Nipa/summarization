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
llm_model_id = "meta-llama/Llama-3.3-70B-Instruct"  # Changed to 70B model
nli_model_id = "roberta-large-mnli"
stats_file = "model_stats.csv"
output_csv = "llama3_70b_zeroshot.csv"  # Updated filename to reflect 70B model

# Entailment thresholds and fallbacks
primary_threshold = 0.6
fallback_threshold = 0.4
min_sentences = 1  # Minimum sentences to include in extractive summary

print("Loading models...")
# ========== 1. Load LLaMA 3.3 70B ==========
llm = LLM(model=llm_model_id, tensor_parallel_size=2, max_model_len=2048, gpu_memory_utilization=0.85, max_num_seqs=4, enforce_eager=True, swap_space=16, quantization="fp8")
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

# ========== 2. Load RoBERTa NLI Model ==========
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_id).half().to("cuda")

print("Loading dataset...")
# ========== 3. Load Dataset ==========
with open("data.pkl", "rb") as f:
    dataset = pickle.load(f)

# ========== 4. Entailment Functions ==========
def get_entailment_probs(premise, hypothesis_list):
    """Calculate entailment probabilities for multiple hypotheses efficiently."""
    if not hypothesis_list or not premise:
        return []
        
    # Handle very long premises by truncation
    if len(premise) > 512:
        premise = premise[:512]
        
    inputs = nli_tokenizer([premise] * len(hypothesis_list), hypothesis_list, 
                         return_tensors="pt", padding=True, truncation=True, 
                         max_length=512).to("cuda")
    
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    
    probs = softmax(logits, dim=1)
    entail_probs = probs[:, 2].tolist()  # Index 2 corresponds to entailment
    return entail_probs

def select_sentences(premise, sentence_objs, primary_threshold=0.6, fallback_threshold=0.4, min_sentences=1):
    """Select sentences based on entailment scores with fallback mechanisms."""
    sentences = [s["sentence"] for s in sentence_objs]
    if not sentences or not premise:
        return ""
        
    # Get entailment probabilities for all sentences
    entail_probs = get_entailment_probs(premise, sentences)
    if not entail_probs:
        return ""
        
    # First try with primary threshold
    selected_indices = [i for i, prob in enumerate(entail_probs) if prob > primary_threshold]
    
    # If we don't have enough sentences, try fallback threshold
    if len(selected_indices) < min_sentences:
        selected_indices = [i for i, prob in enumerate(entail_probs) if prob > fallback_threshold]
    
    # Still not enough? Take top-N sentences
    if len(selected_indices) < min_sentences:
        selected_indices = sorted(range(len(entail_probs)), key=lambda i: entail_probs[i], reverse=True)[:min_sentences]
    
    # Sort by original position to maintain coherence
    selected_indices.sort()  
    
    return " ".join([sentences[i] for i in selected_indices])

# Track seen answers to avoid duplication
seen_answers = set()

# ========== 5. Inference Loop with Metrics ==========
csv_rows = []
predictions, references = [], []

print("Starting inference...")
start_time = time.time()
torch.cuda.reset_peak_memory_stats()

for item in dataset:
    qid = item.get("question_id")
    title = item.get("question_title", "")
    answer = item.get("answer_body", "")
    sentences = item.get("sentences", [])
    
    # Generate or get answer ID
    answer_id = item.get("answer_id", f"{qid}_answer")
    
    # Skip if we've already processed this answer
    if answer_id in seen_answers:
        print(f"Skipping duplicate answer: {answer_id}")
        continue
    seen_answers.add(answer_id)

    # Abstractive Summarization
    # Using a more specific prompt to guide the model
    messages = [
        {"role": "system", "content": "You are an expert at summarizing technical answers from Stack Overflow. Create a brief, accurate summary that captures the essential technical information and any code snippets or key concepts."},
        {"role": "user", "content": f"Question: {title}\n\nAnswer: {answer}\n\nSummary:"}
    ]
    prompt = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    try:
        outputs = llm.generate(prompt, sampling_params)
        abstractive_summary = outputs[0].outputs[0].text.strip()
        
        # Basic cleanup for the abstractive summary
        if abstractive_summary.startswith("Summary:"):
            abstractive_summary = abstractive_summary[8:].strip()
            
        print(f"Generated abstractive summary: {abstractive_summary[:100]}...")
    except Exception as e:
        print(f"Error generating summary for question {qid}: {str(e)}")
        abstractive_summary = f"ERROR: {str(e)}"

    # Extractive Summary via Enhanced NLI
    extractive_summary = select_sentences(
        abstractive_summary, 
        sentences,
        primary_threshold=primary_threshold,
        fallback_threshold=fallback_threshold,
        min_sentences=min_sentences
    )
    
    # If still empty, use the first sentence of the answer as a last resort
    if not extractive_summary and sentences:
        print(f"Warning: No sentences selected for question {qid}. Using first sentence as fallback.")
        extractive_summary = sentences[0]["sentence"]

    # Ground Truth Summary
    ground_truth = " ".join([s["sentence"] for s in sentences if s.get("truth") == 1])

    # Debug info
    print(f"\nProcessed Question ID: {qid}, Answer ID: {answer_id}")
    print(f"Extractive summary length: {len(extractive_summary)} chars")
    print(f"Ground truth length: {len(ground_truth)} chars")
    
    # Save results
    csv_rows.append([qid, answer_id, title, answer, abstractive_summary, extractive_summary, ground_truth])
    if extractive_summary:  # Only include non-empty summaries in evaluation
        predictions.append(extractive_summary)
        references.append(ground_truth)

# ========== 6. Time and GPU Stats ==========
end_time = time.time()
elapsed_time = end_time - start_time
mem_allocated = torch.cuda.max_memory_allocated() / (1024**3)
mem_reserved = torch.cuda.max_memory_reserved() / (1024**3)

# ========== 7. Save Summaries ==========
with open(output_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Question ID", "Answer ID", "Title", "Answer Body", "Abstractive Summary", "Extractive Summary", "Ground Truth"])
    writer.writerows(csv_rows)

print(f"\nSummarization completed. Results saved to {output_csv}")

# ========== 8. Evaluate ROUGE ==========
if predictions and references:
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougel = [], [], []

    for ref, pred in zip(references, predictions):
        scores = rouge.score(ref, pred)
        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rougel.append(scores["rougeL"].fmeasure)

    avg_rouge1 = sum(rouge1)/len(rouge1) if rouge1 else 0
    avg_rouge2 = sum(rouge2)/len(rouge2) if rouge2 else 0
    avg_rougel = sum(rougel)/len(rougel) if rougel else 0

    print("\n?? ROUGE Scores:")
    print(f"ROUGE-1 F1: {avg_rouge1:.4f}")
    print(f"ROUGE-2 F1: {avg_rouge2:.4f}")
    print(f"ROUGE-L F1: {avg_rougel:.4f}")

    # ========== 9. Evaluate BERTScore ==========
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)
    bert_p = P.mean().item()
    bert_r = R.mean().item()
    bert_f1 = F1.mean().item()

    print("\n?? BERTScore:")
    print(f"Precision: {bert_p:.4f}")
    print(f"Recall:    {bert_r:.4f}")
    print(f"F1:        {bert_f1:.4f}")

    # ========== 10. Evaluate BLEU Score ==========
    bleu_scores = []
    smoothie = SmoothingFunction().method4  # Handles short outputs

    for ref, pred in zip(references, predictions):
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
        bleu_scores.append(bleu)

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    print("\nBLEU Score:")
    print(f"BLEU: {avg_bleu:.4f}")

    # ========== 11. Save Evaluation Stats ==========
    try:
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
                f"{avg_bleu:.4f}"
            ])
        print(f"\n?? Stats saved to {stats_file}")
    except Exception as e:
        print(f"Warning: Could not save stats to file: {e}")
else:
    print("\nWarning: No valid predictions generated, skipping evaluation.")

print(f"\nProcessed {len(csv_rows)} answers")
print(f"Time: {elapsed_time:.2f}s | Mem Alloc: {mem_allocated:.2f} GB | Reserved: {mem_reserved:.2f} GB")