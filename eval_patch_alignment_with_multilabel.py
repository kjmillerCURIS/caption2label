import os
import sys
import glob
import pickle
import torch
from eval_utils import DATASET_DICT, evaluate_patch_alignment_with_multilabel
from experiment_params.param_utils import get_params_key
from experiment_params.params import grab_params

#def evaluate_patch_alignment_with_multilabel(dataset_name, dataset_parent_dir, checkpoint, params):


def eval_patch_alignment_with_multilabel(dataset_parent_dir, experiment_dir, is_laclip=False):
    is_laclip = bool(int(is_laclip))

    dataset_names = sorted(DATASET_DICT.keys())
    results_dict = {}
    results_filename = os.path.join(experiment_dir, 'results.pkl')
    if os.path.exists(results_filename):
        with open(results_filename, 'rb') as f:
            results_dict = pickle.load(f)

    checkpoint_filenames = sorted(glob.glob(os.path.join(experiment_dir, 'checkpoints', '*.pth')))
    for checkpoint_filename in checkpoint_filenames:
        checkpoint = torch.load(checkpoint_filename)
        k = (checkpoint['epoch'], checkpoint['step_within_epoch'])
        if k not in results_dict:
            results_dict[k] = {}

        for dataset_name in dataset_names:
            if dataset_name in results_dict[k]:
                print('already did %s, %s, skipping!'%(str(k), dataset_name))
                continue

            p = grab_params(get_params_key(experiment_dir))
            results = evaluate_patch_alignment_with_multilabel(dataset_name, dataset_parent_dir, checkpoint, p, is_laclip=is_laclip)
            results_dict[k][dataset_name] = results
            with open(results_filename, 'wb') as f:
                pickle.dump(results_dict, f)


def usage():
    print('Usage: python eval_patch_alignment_with_multilabel.py <dataset_parent_dir> <experiment_dir> [<is_laclip>=False]')


if __name__ == '__main__':
    eval_patch_alignment_with_multilabel(*(sys.argv[1:]))
