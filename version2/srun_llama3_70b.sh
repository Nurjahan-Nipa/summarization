#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p gpu2
#SBATCH -A loni_csc7135s25
#SBATCH --gres=gpu:2
#SBATCH -o slurm-%j.out-%N
#SBATCH -e slurm-%j.err-%N

# Start logging
date
echo "Starting LLaMA 3.3 70B inference job..."

# Monitor GPU + CPU
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv -l 300 &
vmstat -t 300 &

# (Optional) activate environment
# source ~/.bashrc
# conda activate trial

# Run the model
python llama3_70b.py

echo "Job completed."
date

exit 0

