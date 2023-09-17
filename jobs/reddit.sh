#!/bin/sh
#SBATCH --job-name=asgcn
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH -C A6000
#SBATCH --gres=gpu:1


source activate asgcn
python run_pubmed.py \
--dataset reddit \
--epochs 10 \
--hidden1 64 \
--noattention \
--rank 256 \
--seed $RANDOM
