#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h=!ivcbuyin@scc-304.scc.bu.edu
#$ -l h_rt=11:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate caption2label42
cd ~/data/caption2label
python visualize_training_examples.py ${EXPERIMENT_DIR} ${IS_LACLIP}

