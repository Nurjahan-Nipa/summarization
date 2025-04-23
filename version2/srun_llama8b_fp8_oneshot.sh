#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -N 1               # request one node
#SBATCH -t 12:00:00	        # request two hours
#SBATCH -p gpu2          # in single partition (queue)
#SBATCH -A loni_csc7135s25
#SBATCH --gres=gpu:2
#SBATCH -o slurm-%j.out-%N # optional, name of the stdout, using the job number (%j) and the hostname of the node (%N)
#SBATCH -e slurm-%j.err-%N # optional, name of the stderr, using job and hostname values
# below are job commands
date

echo "Starting LLaMA 3.1 8B fp 8 with few shot"

# Monitor GPU + CPU
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv -l 300 &
vmstat -t 300 &




python llama8b_fp8_oneshot.py
# Set some handy environment variables.
#date
# exit the job
exit 0



