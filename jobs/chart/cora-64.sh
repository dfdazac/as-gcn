#!/bin/sh
#SBATCH --job-name=cora-64
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH -C A4000
#SBATCH --gres=gpu:1
#SBATCH --output=cora_64_%A_%a.out


source activate asgcn
python -u run_pubmed.py \
--dataset cora \
--hidden1 64 \
--noattention \
--rank 64 \
--learning_rate 1e-3 \
--seed $RANDOM
