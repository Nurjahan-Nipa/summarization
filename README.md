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

# Project Structure: HybridSum_v2

This document outlines the folder and file organization of the `HybridSum_v2` project.

```
HybridSum_v2/
│
├── HybridSum_v2.pdf                # Final report
├── HybridSum_v2.pptx               # Project presentation slides
├── README.md                       # Project overview and instructions
├── data.csv                        # CSV dataset
├── data.pkl                        # Pickled version of dataset
├── bashrc_qbd.txt                  # Custom bashrc settings for QBD environment
│
├── llama3_8b_few/                  # Few-shot summarization with LLaMA 3.1 8B
│   ├── llama8b_fp8_fewshot.py      # Few-shot inference script using FP8
│   ├── llama8b_fp8_fewshot.csv     # Output CSV of FP8 few-shot summaries
│   ├── model_stats.csv             # Model statistics for evaluation
│   ├── slurm-265578.err-qbd491     # SLURM error log for job on qbd491
│   ├── slurm-265578.out-qbd491     # SLURM output log for job on qbd491
│   └── srun_llama8b_fp8_fewshot.sh # Job submission script
│
├── llama3_8b_zero/                 # Zero-shot summarization with LLaMA 3.1 8B
│   ├── llama8b_zero_shot.py        # Zero-shot inference script
│   ├── llama3_8b_zeroshot.csv      # Output CSV of zero-shot summaries
│   ├── slurm-265587.err-qbd489     # SLURM error log for job on qbd489
│   ├── slurm-265587.out-qbd489     # SLURM output log for job on qbd489
│   └── srun_llama8b_zero_shot.sh   # Job submission script
│
├── llama3_70b/                     # Summarization with LLaMA 3.3 70B
│   ├── llama8b_fp8_70b.py          # Inference script using LLaMA 3.3 70B and FP8
│   ├── llama3_70b_zeroshot.csv     # Output CSV of 70B zero-shot summaries
│   ├── slurm-265643.err-qbd491     # SLURM error log for job on qbd491
│   ├── slurm-265643.out-qbd491     # SLURM output log for job on qbd491
│   └── srun_llama8b_fp8_70b.sh     # Job submission script
│
│
├── version2/                       # Previous version (legacy scripts and results)
│   ├── llama8b_fp8_fewshot.py      # FP8 few-shot script
│   ├── llama8b_fp8_one_shot.py     # One-shot script
│   ├── llama8b_fp8_twoshot.py      # Two-shot script
│   ├── *.csv                       # Corresponding output files
│   └── slurm-*.err / slurm-*.out   # Logs from prior jobs
│
└── 'private files'/                # Private or sensitive files (folder contains space)
```


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

