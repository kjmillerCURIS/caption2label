#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 12
#$ -l h_rt=2:59:59
#$ -N download_images_cc3m
#$ -j y
#$ -m ea

module load miniconda
conda activate caption2label42
cd ~/data/caption2label
python download_images_cc.py ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_original.csv /net/ivcfs5/mnt/data/nivek/cc3m_images 12

