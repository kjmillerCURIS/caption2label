#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 3
#$ -l h_rt=1:59:59
#$ -N compute_class_prevalence
#$ -j y
#$ -m ea

module load miniconda
conda activate caption2label42
cd ~/data/caption2label
python compute_class_prevalence.py ../vislang-domain-exploration-data/caption2label-data/EvalDatasets ../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_rs0_llama13b_temp0.7_extracted_bulk_adjnoun_prompting ../vislang-domain-exploration-data/caption2label-data/class_prevalence_dict.pkl ../vislang-domain-exploration-data/caption2label-data/class_prevalence_plots/class_prevalence_plot

