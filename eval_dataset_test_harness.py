import os
import sys
import torch
from tqdm import tqdm
from eval_utils import DATASET_DICT

def eval_dataset_test_harness():
    dataset_parent_dir = '/net/ivcfs5/mnt/data/nivek/EvalDatasets'
    context_length = 77
    clip_model_type = 'ViT-B/16'
    for k in sorted(DATASET_DICT.keys()):
        print('test harnessing %s...'%(k))
        image_bases = set([])
        meow = DATASET_DICT[k](dataset_parent_dir, context_length, clip_model_type)
        textsA = meow.get_label_classnames()
        print(textsA.shape)
        textsB = meow.get_templated_classnames()
        print(textsB.shape)
        for idx in tqdm(range(len(meow))):
            datum = meow[idx]
            assert(datum['image'].shape == (3,224,224))
            image_base = meow.get_image_bases(torch.unsqueeze(datum['idx'], dim=0))[0]
            assert(image_base not in image_bases)
            image_bases.add(image_base)


if __name__ == '__main__':
    eval_dataset_test_harness()
