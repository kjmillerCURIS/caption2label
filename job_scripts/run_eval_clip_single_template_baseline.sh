#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 3
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=11:59:59
#$ -N eval_clip_single_template_baseline
#$ -j y
#$ -m ea

module load miniconda
conda activate caption2label42
cd ~/data/caption2label
python eval_clip_single_template_baseline.py ../vislang-domain-exploration-data/caption2label-data/EvalDatasets ../vislang-domain-exploration-data/caption2label-data/Experiments/baselines/experiment_OpenAICheckpointCLIPEnsembleBaseline

