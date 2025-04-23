# 🧠 HybridSum: Hybrid Summarization of Stack Overflow Posts

This repository contains the code, datasets, and SLURM job scripts for **HybridSum**, a hybrid summarization system designed to generate accurate and concise summaries of Stack Overflow answers. The system integrates **abstractive summarization** using LLaMA models with **extractive filtering** using entailment-based sentence selection (RoBERTa-large-MNLI).

---

## 🚀 Key Features

- **Hybrid Two-Stage Pipeline**:
  - Stage 1: LLaMA 3.1/3.3 generates abstractive summaries.
  - Stage 2: RoBERTa-MNLI filters sentences that are entailed by the abstractive output.

- **Few-shot and Zero-shot Prompting**:
  - Efficient prompting strategies are used to guide summarization without fine-tuning.

- **Faithfulness via NLI Filtering**:
  - Ensures generated summaries are grounded in source content.

- **Multi-model Evaluation**:
  - Comparisons between LLaMA-3.3-70B and LLaMA-3.1-8B with performance measured using ROUGE, BLEU, and BERTScore.

---

## 📁 Directory Structure

```
.

├── data.pkl                      # SoSum dataset (Stack Overflow answers)
├── *.py                          # Main LLM execution scripts
├── *.csv                         # Summary results and evaluation outputs
├── slurm-*.out / slurm-*.err     # SLURM job logs
├── srun_*.sh                     # SLURM batch job scripts
└── README.md                     # Project overview
```

---

## 📜 Key Scripts and Outputs

| Script Name               | Description                                 | Sbatch Script                   | Summary / Output File                            |
|--------------------------|---------------------------------------------|----------------------------------|--------------------------------------------------|
| `llama3_70b.py`          | Summarization using LLaMA-3.3 70B           | `srun_llama3_70b.sh`             | `summaries_70b.csv`                              |
| `llama3_8b_base.py`      | Summarization using LLaMA-3.1 8B            | `srun_llama3_8b.sh`              | `llama3_8b_base.csv`                             |
| `llama8b_few_shot_v2.py` | Few-shot summarization using LLaMA-3 8B FP8 | `srun_llama8b_fp8_fewshot.s`     | `slurm-257904.out-qbd486`                        |
| `llama8b_fp8_oneshot.py` | One-shot summarization                      | `srun_llama8b_fp8_oneshot.s`     | `llama8b_fp8_one_shot.csv`                       |
| `llama8b_fp8_twoshot.py` | Two-shot summarization                      | `srun_llama8b_fp8_twoshot.s`     | `llama8b_fp8_twoshot.csv` *(assumed)*            |
| `llama8b_zero_shot.py`   | Zero-shot summarization                     | `srun_llama8b_zero_shot.sh`      | `llama8b_zero_shot_hybrid_20250421_005610.csv`   |

---

## 📊 Evaluation Metrics

- **ROUGE**: N-gram overlap between generated and reference summaries.
- **BLEU**: Precision of n-gram matches (common in translation tasks).
- **BERTScore**: Semantic similarity using contextual embeddings.

---

## 📦 Dataset

- **SoSum**: A curated dataset of ~3,130 Stack Overflow posts including:
  - Conceptual, bug-fix, how-to questions
  - Annotated sentences for summarization relevance

---

## ⚙️ Environment & HPC

- System: LONI HPC (QBD)
- GPUs: 2× A100 80GB
- LLM Inference Engine: `vLLM`
- Quantization: FP8 for efficiency
- Memory Usage: Up to 170GB (8B), 135.8GB ×2 (70B)

---



## 🔮 Future Work

- Fine-tuning NLI on technical Q&A
- Cross-domain extension to GitHub, PubMed, legal texts
- Real-time summarization interface with Streamlit/Gradio

