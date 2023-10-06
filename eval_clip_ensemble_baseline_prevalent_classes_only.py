import os
import sys
import glob
import pickle
import torch
from eval_utils import DATASET_DICT, evaluate_clip_ensemble_baseline
from experiment_params.param_utils import get_params_key
from experiment_params.params import grab_params

#def evaluate_clip_ensemble_baseline(dataset_name, dataset_parent_dir):


def eval_clip_ensemble_baseline_prevalent_classes_only(dataset_parent_dir,experiment_dir,class_prevalence_dict_filename,is_laclip=False):
    is_laclip = bool(int(is_laclip))

    dataset_names = sorted(DATASET_DICT.keys())
    results_dict = {}
    results_filename = os.path.join(experiment_dir, 'results_prevalent_classes_only.pkl')
    if os.path.exists(results_filename):
        with open(results_filename, 'rb') as f:
            results_dict = pickle.load(f)

    for dataset_name in dataset_names:
        if dataset_name in results_dict:
            print('already did %s, skipping!'%(dataset_name))
            continue

        results = evaluate_clip_ensemble_baseline(dataset_name, dataset_parent_dir, class_prevalence_dict_filename=class_prevalence_dict_filename, is_laclip=is_laclip)
        results_dict[dataset_name] = results
        with open(results_filename, 'wb') as f:
            pickle.dump(results_dict, f)


def usage():
    print('Usage: python eval_clip_ensemble_baseline_prevalent_classes_only.py <dataset_parent_dir> <experiment_dir> <class_prevalence_dict_filename> [<is_laclip>=False]')


if __name__ == '__main__':
    eval_clip_ensemble_baseline_prevalent_classes_only(*(sys.argv[1:]))
