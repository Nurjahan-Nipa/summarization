# 🦙 LLaMA Summarization Experiments

This repository contains source code, job scripts, datasets, and outputs related to experiments on automatic summarization using LLaMA models (8B and 70B). The experiments involve few-shot, zero-shot, one-shot, and hybrid prompting strategies on a Stack Overflow summarization dataset. SLURM job logs are also included from HPC runs (e.g., QBD cluster).

---

## 📁 Project Structure

```

├── llama3_*.py                    # LLaMA-3 model inference scripts
├── llama8b_fp8_*.py               # LLaMA-3 8B FP8 inference scripts
├── *.csv                          # Model predictions, summaries, and evaluation metrics
├── model_stats*.csv               # ROUGE/BERTScore/BLEU results
├── slurm-*.err-qbd*               # SLURM error logs
├── slurm-*.out-qbd*               # SLURM output logs
├── srun_*.sh                      # SLURM job scripts to launch experiments
└── readme.md                      # This file
```

---

## 🧠 Key Scripts

| Script Name                     | Description                                |Sbatch script                   |Summary               | 
|--------------------------------|---------------------------------------------|............................... |.......................|
| `llama3_70b.py`                | Summarization using LLaMA-3.3 70B           |                                |summaries_70b.csv       |
| `llama3_8b_base.py`            | Summarization using LLaMA-3.1 8B            | |
| 'llama8b_few_shot_v2.py'       | Few-shot summarization using LLaMA-3 8B FP8 | srun_llama8b_few_shot_v2.py    |output file: slurm-257904.out-qbd486 |
| `llama8b_fp8_oneshot.py`       | One-shot summarization                     | | |
| `llama8b_fp8_twoshot.py`       | Two-shot summarization                     | | |
| `llama8b_zero_shot.py`         | Zero-shot summarization                    | | |


---

## 🧪 Running Experiments

Use SLURM to submit jobs to the HPC cluster.

```bash
sbatch srun_llama3_70b.sh
sbatch srun_llama8b_few_shot_v2.sh
```

You can also run scripts directly (on smaller models or in debug mode):

```bash
python llama8b_few_shot_v2.py
```

---

## 📊 Outputs & Logs

- `*.csv`: Summarized outputs, filtered summaries, and evaluation results.
- `model_stats*.csv`: Contains evaluation metrics (ROUGE, BLEU, BERTScore).
- `slurm-*.out`, `slurm-*.err`: SLURM logs for each experiment.

---

## 📦 Dataset

- `data.pkl`: A pickle file containing Stack Overflow posts with fields like:
  - `question_title`, `answer_body`, `question_tags`, `sentences`

---

## ⚙️ Environment

Use a Python virtual environment using (https://github.com/pyenv/pyenv)  and then pyenv virtualenv using (https://github.com/pyenv/pyenv-virtualenv)

Activate your environment:
```bash
source ~/llama-env/bin/activate
```

Install required packages:
```bash
pip install -r requirements.txt
```

---

## 📬 Contact

Maintained by **Nurjahan Nipa**  
📧 Email: [nurja1@lsu.edu](mailto:nurja1@lsu.edu)

---

## 📌 Notes

- Designed for use with QBD (LONI HPC) and similar clusters.
- Ensure `transformers`, `torch`, `flash-attention`, and `datasets` are properly installed before running.
