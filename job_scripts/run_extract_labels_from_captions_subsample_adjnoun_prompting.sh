#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=2:59:59
#$ -N extract_labels_from_captions_subsample_adjnoun_prompting
#$ -j y
#$ -m ea

module load miniconda
conda activate caption2label42
cd ~/data/caption2label
#python extract_labels_from_captions_subsample_adjnoun_prompting.py ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_original.csv ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_eval_rs0_keylist.pkl ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_support_rs0_gt_label_dict.pkl 13B 0.7 ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_rs0_llama13b_temp0.7_extracted_subsample_adjnoun_prompting.pkl
python extract_labels_from_captions_subsample_adjnoun_prompting.py ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_original.csv ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_eval_rs0_keylist.pkl ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_support_rs0_gt_label_dict.pkl 13B 0.3 ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_rs0_llama13b_temp0.3_extracted_subsample_adjnoun_prompting.pkl
python extract_labels_from_captions_subsample_adjnoun_prompting.py ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_original.csv ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_eval_rs0_keylist.pkl ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_support_rs0_gt_label_dict.pkl 13B 0.0 ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_rs0_llama13b_temp0.0_extracted_subsample_adjnoun_prompting.pkl

