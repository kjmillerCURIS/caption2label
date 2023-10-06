import os
import sys
import numpy as np
import pickle
import random
import torch
from tqdm import tqdm
from experiment_params.param_utils import get_params_key
from experiment_params.params import grab_params
from checkpoint_utils import load_patch_alignment_model_from_checkpoint
from load_training_examples_for_visualization import OUTPUT_FILENAME as TRAINING_DATA_FILENAME
from vis_utils import plot_one_training_example


#def plot_one_training_example(image_datum, text_datum, is_positive_pair, params, model, plot_prefix):


RANDOM_SEED = 0
IMAGE_DROP_RATE = 6


#return image_datum_list, text_datum_list, neg_pair_mapping
def load_training_data(params):
    p = params
    random.seed(RANDOM_SEED)
    examples = torch.load(TRAINING_DATA_FILENAME)
    image_datum_list = [example[0][p.image_aug_type] for example in examples]
    text_datum_list = [example[1] for example in examples]
    neg_pair_mapping = [random.choice([j for j in range(len(examples)) if j != i]) for i in range(len(examples))]
    return image_datum_list, text_datum_list, neg_pair_mapping


def get_plot_prefix(experiment_dir, params_key, checkpoint_suffix, image_datum, is_positive_pair, is_laclip=False):
    epoch_str = 'epoch' + checkpoint_suffix.split('-')[0] if checkpoint_suffix != 'FINAL' else 'FINAL'
#    epoch_str = 'epoch' + checkpoint_suffix if checkpoint_suffix != 'FINAL' else 'FINAL'
    plot_dir = os.path.join(experiment_dir, 'train_vis', epoch_str, {True : 'pos_pairs', False : 'neg_pairs'}[is_positive_pair])
    plot_prefix_base = params_key+'-'+{True:'LaCLIP',False:''}[is_laclip]+'-'+epoch_str+'-'+'img%09d'%(image_datum['idx'])
    return os.path.join(plot_dir, plot_prefix_base)


def visualize_training_examples(experiment_dir, is_laclip=False):
    is_laclip = bool(int(is_laclip))

    params_key = get_params_key(experiment_dir)
    p = grab_params(params_key)
    image_datum_list, text_datum_list, neg_pair_mapping = load_training_data(p)
    for checkpoint_suffix in tqdm(reversed(['%03d-000000000'%(i) for i in range(10)] + ['FINAL'])):
#    for checkpoint_suffix in tqdm(['000-000001334','000-000002667','000-000004000','001-000001334','001-000002667','001-000004000']):
        checkpoint_filename = os.path.join(experiment_dir, 'checkpoints', 'checkpoint-' + checkpoint_suffix + '.pth')
        checkpoint = torch.load(checkpoint_filename)
        print('loading model...')
        model = load_patch_alignment_model_from_checkpoint(p, checkpoint, is_laclip=is_laclip)
        print('done loading model')
        for is_positive_pair in [True, False]:
            for i, image_datum in tqdm(enumerate(image_datum_list)):
                if i % IMAGE_DROP_RATE != 0:
                    continue

                if is_positive_pair:
                    text_datum = text_datum_list[i]
                else:
                    text_datum = text_datum_list[neg_pair_mapping[i]]

                plot_prefix = get_plot_prefix(experiment_dir, params_key, checkpoint_suffix, image_datum, is_positive_pair, is_laclip=is_laclip)
                os.makedirs(os.path.dirname(plot_prefix), exist_ok=True)
                plot_one_training_example(image_datum, text_datum, is_positive_pair, p, model, plot_prefix)


def usage():
    print('Usage: python visualize_training_examples.py <experiment_dir> [<is_laclip>=False]')


if __name__ == '__main__':
    visualize_training_examples(*(sys.argv[1:]))
