#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=32G
#$ -l h_rt=23:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate caption2label42
cd ~/data/caption2label
python extract_labels_from_captions_bulk.py ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_original.csv ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_support_rs0_gt_label_dict.pkl ${START_INDEX} 16 ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_rs0_llama13b_temp0.7_extracted_bulk_adjnoun_prompting_part${START_INDEX}.pkl

