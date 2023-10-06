#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 3
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=11:59:59
#$ -l h=!ivcbuyin@scc-304.scc.bu.edu
#$ -j y
#$ -m ea

module load miniconda
conda activate caption2label42
cd ~/data/caption2label
python eval_patch_alignment_with_multilabel_prevalent_classes_only.py ../vislang-domain-exploration-data/caption2label-data/EvalDatasets ${EXPERIMENT_DIR} ../vislang-domain-exploration-data/caption2label-data/class_prevalence_dict.pkl ${IS_LACLIP}

