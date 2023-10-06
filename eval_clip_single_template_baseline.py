import os
import sys
import glob
import pickle
import torch
from eval_utils import DATASET_DICT, evaluate_clip_single_template_baseline
from experiment_params.param_utils import get_params_key
from experiment_params.params import grab_params

#def evaluate_clip_single_template_baseline(dataset_name, dataset_parent_dir):


def eval_clip_single_template_baseline(dataset_parent_dir, experiment_dir, is_laclip=False):
    is_laclip = bool(int(is_laclip))

    dataset_names = sorted(DATASET_DICT.keys())
    results_dict = {}
    results_filename = os.path.join(experiment_dir, 'results_single_template.pkl')
    if os.path.exists(results_filename):
        with open(results_filename, 'rb') as f:
            results_dict = pickle.load(f)

    for dataset_name in dataset_names:
        if dataset_name in results_dict:
            print('already did %s, skipping!'%(dataset_name))
            continue

        results = evaluate_clip_single_template_baseline(dataset_name, dataset_parent_dir, is_laclip=is_laclip)
        results_dict[dataset_name] = results
        with open(results_filename, 'wb') as f:
            pickle.dump(results_dict, f)


def usage():
    print('Usage: python eval_clip_single_template_baseline.py <dataset_parent_dir> <experiment_dir> [<is_laclip>=False]')


if __name__ == '__main__':
    eval_clip_single_template_baseline(*(sys.argv[1:]))
