import os
import sys
import numpy as np
import random
import torch
from tqdm import tqdm
from experiment_params.params import grab_params
from image_text_dataset_multilabel import ImageTextDatasetMultilabel


NUM_EXAMPLES = 300
RANDOM_SEED = 0
PARAMS_KEY_DICT = {'simclr_A' : 'PatchAlignmentMultilabelParams', 'clip_A' : 'PatchAlignmentMultilabelClipAugParams'}
TEXT_FILENAME_PREFIX = '../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_rs0_llama13b_temp0.7_extracted_bulk_adjnoun_prompting'
IMAGE_DIR = '../vislang-domain-exploration-data/caption2label-data/cc3m_images'
CAPTION_CSV_FILENAME = '../vislang-domain-exploration-data/caption2label-data/caption-processing/cc3m_original.csv'
OUTPUT_FILENAME = '../vislang-domain-exploration-data/caption2label-data/cc3m_training_examples.pth'


def load_training_examples_for_visualization():
    dataset_dict = {}
    for aug_type in sorted(PARAMS_KEY_DICT.keys()):
        p = grab_params(PARAMS_KEY_DICT[aug_type])
        assert(p.image_aug_type == aug_type)
        dataset_dict[aug_type] = ImageTextDatasetMultilabel(IMAGE_DIR, TEXT_FILENAME_PREFIX, p, is_for_vis=True, caption_csv_filename=CAPTION_CSV_FILENAME)

    total = len(dataset_dict[sorted(dataset_dict.keys())[0]])
    assert(all([len(dataset_dict[aug_type]) == total for aug_type in sorted(dataset_dict.keys())]))
    random.seed(RANDOM_SEED)
    idxs = random.sample(range(total), NUM_EXAMPLES)
    examples = []
    for idx in tqdm(idxs):
        image_datum = {}
        for aug_type in sorted(dataset_dict.keys()):
            image_datum[aug_type], text_datum = dataset_dict[aug_type][idx]

        examples.append((image_datum, text_datum))

    torch.save(examples, OUTPUT_FILENAME)


if __name__ == '__main__':
    load_training_examples_for_visualization()
