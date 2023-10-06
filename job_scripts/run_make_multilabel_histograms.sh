#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=0:59:59
#$ -N make_multilabel_histograms
#$ -j y
#$ -m ea

module load miniconda
conda activate caption2label42
cd ~/data/caption2label
python make_multilabel_histograms.py ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_rs0_llama13b_temp0.7_extracted_bulk_adjnoun_prompting num_labels_histogram.png num_tokens_histogram.png

