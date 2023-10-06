#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 3
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=24G
#$ -l h_rt=2:59:59
#$ -N llama_example
#$ -j y
#$ -m ea

module load miniconda
conda activate caption2label42
cd ~/data/llama
torchrun --nproc_per_node 1 example.py --ckpt_dir ~/data/vislang-domain-exploration-data/caption2label-data/llama-models/llama-7b-hf --tokenizer_path ~/data/vislang-domain-exploration-data/caption2label-data/llama-models/llama-7b-hf/tokenizer.model

