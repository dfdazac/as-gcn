#!/bin/sh
#SBATCH --job-name=arxiv-256
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH -C A4000
#SBATCH --gres=gpu:1
#SBATCH --output=arxiv_256_%A_%a.out


source activate asgcn
python -u run_pubmed.py \
--dataset arxiv \
--hidden1 64 \
--noattention \
--rank 256 \
--learning_rate 1e-3 \
--seed $RANDOM
