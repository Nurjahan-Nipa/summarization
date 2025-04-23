import pickle
import csv
import torch
import time
import subprocess
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ========== CONFIGURATION ==========
llm_model_id = "meta-llama/Llama-3.3-70B-Instruct"
nli_model_id = "roberta-large-mnli"
output_file = "summaries_70b.csv"
stats_file = "model_stats.csv"

# Configure logging
logging.basicConfig(filename='summarization.log', level=logging.INFO)

# ========== 1. Load LLaMA ==========
try:
    llm = LLM(
        model=llm_model_id,
        tensor_parallel_size=2,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        max_num_seqs=4,
        enforce_eager=True,
        swap_space=16,
        quantization="fp8"
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
except Exception as e:
    logging.error(f"LLaMA initialization failed: {str(e)}")
    raise

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.9,
    max_tokens=128,
    skip_special_tokens=True
)

# ========== 2. Load RoBERTa NLI ==========
try:
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_id).half().to('cuda')
except Exception as e:
    logging.error(f"NLI model loading failed: {str(e)}")
    raise

# ========== 3. Entailment Function ==========
def entails(premise, hypothesis, threshold=0.6):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to('cuda')
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = softmax(logits, dim=1)
    return probs[0][2].item() > threshold

# ========== 4. Truncate for LLaMA ==========
def safe_truncate(text, tokenizer, max_tokens=1600):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

# ========== 5. Main Processing ==========
def process_item(item):
    try:
        question = item["question_title"]
        answer = item["answer_body"]
        answer_truncated = safe_truncate(answer, llm_tokenizer)

        messages = [
            {"role": "system", "content": "You are an expert summarizer. Your task is to generate a brief and accurate summary of the provided Stack Overflow answer, based on the question context."},
            {"role": "user", "content": f"Question: {question}\nAnswer: {answer_truncated}\nSummary:"}
        ]
        prompt = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        outputs = llm.generate(prompt, sampling_params)
        abstractive = outputs[0].outputs[0].text.strip()

        extracted = [
            s["sentence"] for s in item["sentences"]
            if entails(abstractive, s["sentence"])
        ]

        return [
            item["question_id"], item["question_title"], item["answer_body"],
            abstractive, " ".join(extracted),
            " ".join(s["sentence"] for s in item["sentences"] if s.get("truth") == 1)
        ]
    except Exception as e:
        logging.warning(f"Failed on QID {item.get('question_id')}: {str(e)}")
        return [
            item.get("question_id"),
            item.get("question_title", ""),
            item.get("answer_body", ""),
            f"ERROR: {str(e)}", "", ""
        ]

# ========== 6. MAIN EXECUTION ==========
if __name__ == "__main__":
    # Load data
    try:
        with open("data.pkl", "rb") as f:
            dataset = pickle.load(f)
    except Exception as e:
        logging.critical(f"Data loading failed: {str(e)}")
        raise

    # Start timing
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    predictions, references, csv_rows = [], [], []

    for item in tqdm(dataset, desc="Processing Questions"):
        result = process_item(item)
        csv_rows.append(result)
        if result[4]:
            predictions.append(result[4])
            references.append(result[5])

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # GPU memory usage
    mem_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    mem_reserved = torch.cuda.max_memory_reserved() / (1024**3)

    # Save summaries to CSV
    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Question ID", "Title", "Answer Body",
            "Abstractive Summary", "Extractive Summary", "Ground Truth"
        ])
        writer.writerows(csv_rows)

    # ========== Evaluation ==========
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {k: [] for k in ['rouge1', 'rouge2', 'rougeL']}

    for ref, pred in zip(references, predictions):
        scores = rouge.score(ref, pred)
        for k in rouge_scores:
            rouge_scores[k].append(scores[k].fmeasure)

    print("\n📊 ROUGE Scores:")
    for metric, values in rouge_scores.items():
        print(f"{metric}: {sum(values)/len(values):.4f}")

    print("\n📊 BERTScore:")
    P, R, F1 = bert_score(predictions, references, lang="en")
    print(f"Precision: {P.mean():.4f}")
    print(f"Recall:    {R.mean():.4f}")
    print(f"F1:        {F1.mean():.4f}")

    # ========== BLEU ==========
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    for ref, pred in zip(references, predictions):
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
        bleu_scores.append(bleu)

    print("\n📊 BLEU Score:")
    print(f"BLEU: {sum(bleu_scores)/len(bleu_scores):.4f}")

    # ========== Report Inference Stats ==========
    print("\n🕒 Inference Time: {:.2f} seconds".format(elapsed_time))
    print("💾 Max memory allocated: {:.2f} GB".format(mem_allocated))
    print("💾 Max memory reserved:  {:.2f} GB".format(mem_reserved))

    # ========== Save Stats ==========
    with open(stats_file, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            llm_model_id,
            f"{elapsed_time:.2f}",
            f"{mem_allocated:.2f}",
            f"{mem_reserved:.2f}",
            f"{sum(rouge_scores['rouge1'])/len(rouge_scores['rouge1']):.4f}",
            f"{sum(rouge_scores['rouge2'])/len(rouge_scores['rouge2']):.4f}",
            f"{sum(rouge_scores['rougeL'])/len(rouge_scores['rougeL']):.4f}",
            f"{F1.mean():.4f}",
            f"{sum(bleu_scores)/len(bleu_scores):.4f}"
        ])

