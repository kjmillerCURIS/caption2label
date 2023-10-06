#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=11:59:59
#$ -N setup_imagenet1k_eval_dataset
#$ -j y
#$ -m ea

module load miniconda
conda activate caption2label42
cd ~/data/caption2label
python setup_imagenet1k_eval_dataset.py /net/ivcfs5/mnt/data/nivek/EvalDatasets

