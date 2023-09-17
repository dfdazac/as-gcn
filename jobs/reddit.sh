#!/bin/sh
#SBATCH --job-name=asgcn
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH -C A6000
#SBATCH --gres=gpu:1


source activate asgcn
python -u run_pubmed.py \
--dataset reddit \
--hidden1 64 \
--noattention \
--rank 256 \
--learning_rate 1e-2 \
--seed $RANDOM
