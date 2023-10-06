#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 5
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=32G
#$ -l h_rt=47:59:59
#$ -N train_clip_with_patch_alignment_and_multilabel_embedder_only
#$ -j y
#$ -m ea

module load miniconda
conda activate caption2label42
cd ~/data/caption2label
python train_clip_with_patch_alignment_and_multilabel.py ../vislang-domain-exploration-data/caption2label-data/cc3m_images ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_rs0_llama13b_temp0.7_extracted_bulk_adjnoun_prompting ../vislang-domain-exploration-data/caption2label-data/Experiments/cc3m/experiment_PatchAlignmentMultilabelEmbedderOnly

