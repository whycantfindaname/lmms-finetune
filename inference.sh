#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=28
#SBATCH --partition=gpu
#SBATCH --exclude=gpu19,gpu3,gpu8,gpu14,gpu17
#SBATCH --job-name=lmms-finetune
#SBATCH --output=/home/u9920230028/lmms-finetune/testbug/inference_job_output.txt
#SBATCH --error=/home/u9920230028/lmms-finetune/testbug/inference_job_error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jasonliaonk21@gmail.com
python inference.py