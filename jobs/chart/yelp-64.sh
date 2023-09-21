#!/bin/sh
#SBATCH --job-name=yelp-64
#SBATCH --time=03:00:00
#SBATCH -N 1
#SBATCH -C A4000
#SBATCH --gres=gpu:1
#SBATCH --output=yelp_64_%A_%a.out


source activate asgcn
python -u run_pubmed.py \
--dataset yelp \
--hidden1 64 \
--noattention \
--rank 64 \
--learning_rate 1e-3 \
--objective multilabel \
--seed $RANDOM
