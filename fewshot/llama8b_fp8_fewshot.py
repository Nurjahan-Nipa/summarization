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
output_csv = "fewshot.csv"
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
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=512)  # Increased max_tokens to handle code better

nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_id).half().to("cuda")

print("Loading dataset...")
with open("data.pkl", "rb") as f:
    dataset = pickle.load(f)

# Significantly revised prompt with clearer separation between examples and task
few_shot_prompt = """You are an expert summarizer for Stack Overflow answers. Your task is to generate CONCISE summaries of technical answers that capture the essential information.

IMPORTANT INSTRUCTIONS:
1. Include key code snippets in your summary when present
2. Focus on the answer's main technical solution
3. Be brief but complete
4. DO NOT repeat the examples in your output
5. DO NOT include phrases like "Here's how to..." or "The solution is..."
6. Start directly with the technical explanation

Here are some examples of good summaries:
"""

# Use fixed examples that are clearly separated from the task
sample_examples = [
    {
        "question": "How do I calculate someone's age in C#?",
        "answer": "Many years ago, to provide an age calculator gimmick on my website, I wrote a function to calculate age to a fraction. This is a quick port of that function to C# (from the PHP version).",
        "summary": "A function to calculate age to a fraction, ported from PHP to C#."
    },
    {
        "question": "What are MVP and MVC and what is the difference?",
        "answer": "MVP = Model-View-Presenter, MVC = Model-View-Controller. Both presentation patterns separate domain objects, UI, and behavior. The main difference is that in MVC the Model updates the View directly.",
        "summary": "MVP = Model-View-Presenter, MVC = Model-View-Controller. The key difference is MVC allows the model to update the view directly."
    }
]

for i, ex in enumerate(sample_examples):
    q = ex["question"]
    a = ex["answer"] 
    s = ex["summary"]
    few_shot_prompt += f"EXAMPLE {i+1}:\nQuestion: {q}\nAnswer: {a}\nSummary: {s}\n\n"

# Add a very clear separator to indicate the end of examples
few_shot_prompt += "END OF EXAMPLES\n\n"
few_shot_prompt += "NOW SUMMARIZE THE FOLLOWING ANSWER:\n"

smoothie = SmoothingFunction().method4

# Improved entailment scoring with better handling of empty inputs
def get_entail_scores(premise, candidate_sents):
    if not candidate_sents or not premise:  # Better handling of empty inputs
        return []
        
    # Handle very long premises by truncating
    if len(premise) > 512:
        premise = premise[:512]
        
    inputs = nli_tokenizer([premise] * len(candidate_sents), candidate_sents, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = softmax(logits, dim=1)
    entail_scores = probs[:, 2].tolist()
    return entail_scores

# Improved selection with better handling of sentence ordering
def select_extractive(summary, sentences):
    candidates = [s["sentence"] for s in sentences]
    if not candidates or not summary:
        return ""

    scores = get_entail_scores(summary, candidates)
    
    if not scores:
        return ""

    if top_k:
        # Get top_k sentences but preserve original order
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        # Sort indices to maintain original sentence order
        top_indices = sorted(top_indices)
        return " ".join([candidates[i] for i in top_indices])
    else:
        # Preserve original order for threshold-based selection
        selected = [(i, s) for i, (s, score) in enumerate(zip(candidates, scores)) if score > entail_threshold]
        selected.sort(key=lambda x: x[0])  # Sort by original position
        return " ".join([s for _, s in selected])

# Track seen answers to avoid duplication
seen_answers = set()  # Initialize the seen_answers set before using it

start = time.time()
torch.cuda.reset_peak_memory_stats()
csv_rows, predictions, references = [], [], []

# If there's a specific reason to skip the first 3 items, add a comment explaining why
for item in dataset[3:]:  # Starting from index 3 (skipping first 3 items)
    qid = item["question_id"]
    title = item["question_title"]
    answer = item["answer_body"]
    sentences = item["sentences"]
    
    # Generate answer ID - use answer_id if available, otherwise create a unique identifier
    answer_id = item.get("answer_id", f"{qid}_answer")
    
    # Check if we've already processed this specific answer
    if answer_id in seen_answers:
        print(f"Skipping duplicate answer: {answer_id}")
        continue
    seen_answers.add(answer_id)

    abstractive = ""
    if mode in ["abstractive", "hybrid"]:
        # Create a prompt with a clear task structure
        task_prompt = f"Question: {title}\nAnswer: {answer}\n\nYour summary:"
        messages = [{"role": "user", "content": few_shot_prompt + task_prompt}]
        prompt = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        try:
            # Generate and post-process to remove any example artifacts
            outputs = llm.generate(prompt, sampling_params)
            raw_output = outputs[0].outputs[0].text.strip()
            
            # Clean up the output to remove any artifacts from prompt examples
            # Remove any lines that contain "Example" or "Question:" (unless it's the first line)
            cleaned_lines = []
            for line in raw_output.split('\n'):
                if ("EXAMPLE" in line.upper() or 
                    "END OF EXAMPLES" in line.upper() or
                    "### " in line or
                    (("Question:" in line or "Answer:" in line) and len(cleaned_lines) > 0)):
                    continue
                cleaned_lines.append(line)
            
            abstractive = '\n'.join(cleaned_lines).strip()
            
            # Final safety check - if we still have prompt artifacts, take just the first paragraph
            if "EXAMPLE" in abstractive.upper() or "END OF EXAMPLES" in abstractive.upper():
                paragraphs = [p for p in abstractive.split('\n\n') if p.strip()]
                if paragraphs:
                    abstractive = paragraphs[0].strip()
            
            print(f"Raw output: {raw_output[:100]}...")
            print(f"Cleaned output: {abstractive[:100]}...")
            
        except Exception as e:
            print(f"Error generating abstractive summary for question {qid}: {str(e)}")
            abstractive = f"ERROR: {str(e)}"

    extractive = ""
    if mode in ["extractive", "hybrid"] and abstractive:
        # Select sentences from the ORIGINAL answer, not from the abstractive summary
        # This will prevent extracting parts of the prompt templates
        extractive = select_extractive(abstractive, sentences).strip()
        
        # If no sentences were selected, try using different entailment thresholds
        if not extractive:
            # Try a lower threshold to be more inclusive
            backup_scores = get_entail_scores(abstractive, [s["sentence"] for s in sentences])
            if backup_scores:
                # Select sentences with scores above a lower threshold
                candidates = [s["sentence"] for s in sentences]
                backup_threshold = 0.4  # Lower threshold for backup
                selected = [(i, s) for i, (s, score) in enumerate(zip(candidates, backup_scores)) if score > backup_threshold]
                selected.sort(key=lambda x: x[0])  # Sort by original position
                extractive = " ".join([s for _, s in selected])

    # Better hybrid mode handling
    if mode == "hybrid":
        if not extractive and abstractive:
            # If extractive is empty but abstractive exists, use abstractive
            # But first check that abstractive doesn't contain prompt artifacts
            if ("EXAMPLE" not in abstractive.upper() and 
                "END OF EXAMPLES" not in abstractive.upper() and
                "### " not in abstractive):
                extractive = abstractive
            else:
                # If abstractive has artifacts, try to extract just the first paragraph
                paragraphs = [p for p in abstractive.split('\n\n') if p.strip()]
                if paragraphs:
                    extractive = paragraphs[0].strip()
        elif extractive:
            # If we have both, but extractive is very short, use a combination
            if len(extractive.split()) < 5 and len(abstractive.split()) > len(extractive.split()):
                if ("EXAMPLE" not in abstractive.upper() and 
                    "END OF EXAMPLES" not in abstractive.upper() and
                    "### " not in abstractive):
                    extractive = abstractive
                
    # Final cleanup of the extractive summary
    if extractive:
        # Remove any remaining prompt artifacts
        lines_to_remove = [
            "EXAMPLE", "END OF EXAMPLES", "### Example", 
            "Question:", "Answer:", "Summary:"
        ]
        
        # Remove lines with prompt artifacts
        clean_lines = []
        for line in extractive.split('\n'):
            if not any(artifact in line for artifact in lines_to_remove):
                clean_lines.append(line)
        
        extractive = '\n'.join(clean_lines)
        
        # Ensure we don't have duplicate sentences
        unique_sentences = []
        for sent in extractive.split('. '):
            sent = sent.strip()
            if sent and sent not in unique_sentences:
                unique_sentences.append(sent)
        
        extractive = '. '.join(unique_sentences)
        
        # Final validation - ensure we're not including any prompt artifacts
        if any(artifact in extractive for artifact in ["EXAMPLE", "END OF EXAMPLES", "###"]):
            # Last resort - try to take just the first sentence
            first_sentence = extractive.split('.')[0].strip()
            if len(first_sentence) > 10:  # Ensure it's not too short
                extractive = first_sentence + "."

    ground_truth = " ".join([s["sentence"] for s in sentences if s.get("truth") == 1])
    
    # Debugging information
    print(f"\nProcessing Question ID: {qid}")
    print(f"Abstractive: {abstractive[:100]}...")
    print(f"Extractive: {extractive[:100]}...")
    print(f"Ground Truth: {ground_truth[:100]}...")
    print(f"Answer length: {len(answer)} chars, {len(answer.split())} words")
    print(f"Number of sentences: {len(sentences)}")
    print(f"Ground truth length: {len(ground_truth)} chars")
    
    csv_rows.append([qid, answer_id, title, answer, abstractive, extractive, ground_truth])  # Added answer_id to CSV row
    predictions.append(extractive)
    references.append(ground_truth)

# Save outputs
with open(output_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Question ID", "Answer ID", "Title", "Answer Body", "Abstractive Summary", "Extractive Summary", "Ground Truth"])
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
print(f"BERTScore:")
print(f"  Precision: {P.mean().item():.4f}")
print(f"  Recall:    {R.mean().item():.4f}")
print(f"  F1:        {F1.mean().item():.4f}")
print(f"BLEU: {sum(bleu)/len(bleu):.4f}")
print(f"\nProcessed {len(csv_rows)} answers from {len(set(row[0] for row in csv_rows))} questions")
print(f"Time: {elapsed:.2f}s | Mem Alloc: {mem_alloc:.2f} GB | Reserved: {mem_reserved:.2f} GB")