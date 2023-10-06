import os
import sys
import clip
import glob
import numpy as np
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose,RandomResizedCrop,RandomHorizontalFlip,RandomApply,ColorJitter,RandomGrayscale,GaussianBlur
from torchvision.transforms.functional import InterpolationMode
from torch.nn import ModuleList
from tqdm import tqdm
from hand_eval import sanitize_labels
from write_to_log_file import write_to_log_file
from hand_label_captions import load_captions


class ImageTextDatasetMultilabel(Dataset):

    def __init__(self, image_dir, text_filename_prefix, params, is_for_vis=False, caption_csv_filename=None):
        p = params
        self.is_for_vis = is_for_vis
        write_to_log_file('_init_image_data...')
        self._init_image_data(image_dir)
        write_to_log_file('_init_text_data...')
        self._init_text_data(text_filename_prefix)
        self.key_list = sorted(self.image_filename_dict.keys() & self.labels_dict.keys())
        write_to_log_file('found %d image-text pairs'%(len(self.key_list)))
        write_to_log_file('_init_image_transforms...')
        self._init_image_transforms(p)
        self.max_num_tokens = p.max_num_tokens
        self.max_num_labels = p.max_num_labels
        if self.is_for_vis:
            self.unnorm_scale = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
            self.unnorm_bias = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
            self.caption_dict = load_captions(caption_csv_filename)

    #self.image_filename_dict will map from key to image_filename (full)
    def _init_image_data(self, image_dir):
        with open(os.path.join(image_dir, 'image_filename_dict.pkl'), 'rb') as f:
            self.image_filename_dict = pickle.load(f)

        write_to_log_file('found %d images'%(len(self.image_filename_dict)))

    def _init_text_data(self, text_filename_prefix):
        self.labels_dict = {}
        text_filenames = sorted(glob.glob(text_filename_prefix + '_part*.pkl'))
        print('found %d parts for text'%(len(text_filenames)))
        for text_filename in text_filenames:
            with open(text_filename, 'rb') as f:
                text_dict = pickle.load(f)

            for k in sorted(text_dict.keys()):
                assert(k not in self.labels_dict)
                my_labels = sorted(set(sanitize_labels(text_dict[k]['labels'])))
                if len(my_labels) > 0:
                    self.labels_dict[k] = my_labels
                else:
                    write_to_log_file('skipping "%s" because labels is empty'%(k))

    #self.image_transforms will be a transform that composes both the augmentations and the CLIP preprocessing
    def _init_image_transforms(self, params):
        p = params
        _, base_transforms = clip.load(p.clip_model_type, device='cpu')
        if p.image_aug_type == 'simclr_A':
            aug_transforms = Compose([
                                    RandomResizedCrop(size=224),
                                    RandomHorizontalFlip(p=0.5),
                                    RandomApply(ModuleList([ColorJitter(0.8, 0.8, 0.8, 0.2)]), p=0.8),
                                    RandomGrayscale(p=0.2),
                                    RandomApply(ModuleList([GaussianBlur(kernel_size=23)]), p=0.5)
                                ])
        elif p.image_aug_type == 'clip_A':
            aug_transforms = Compose([RandomResizedCrop(size=224, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC)])
        else:
            assert(False)

        self.image_transforms = Compose([aug_transforms, base_transforms])

    def _get_image(self, k):
        image_filename = self.image_filename_dict[k]
        image = Image.open(image_filename)
        image = self.image_transforms(image) #aug and preprocess
        return image

    def _get_text(self, k):
        labels = self.labels_dict[k]
        text = clip.tokenize(labels, context_length=self.max_num_tokens, truncate=True)
        assert(len(text.shape) == 2)
        assert(text.shape[1] == self.max_num_tokens)
        if text.shape[0] < self.max_num_labels:
            text_mask = np.concatenate((np.ones(text.shape[0]),np.zeros(self.max_num_labels-text.shape[0])))
            text_mask = torch.tensor(text_mask, dtype=torch.int32)
            padding = clip.tokenize(['meow'] * (self.max_num_labels-text.shape[0]), context_length=self.max_num_tokens, truncate=True)
            text = torch.cat([text, padding], dim=0)
            assert(text.shape == (self.max_num_labels, self.max_num_tokens))
            return text, text_mask
        else: #either too many labels, or just the right amount
            return text[:self.max_num_labels], torch.tensor(np.ones(self.max_num_labels), dtype=torch.int32)

    def __getitem__(self, idx):
        k = self.key_list[idx]
        image = self._get_image(k)
        text, text_mask = self._get_text(k)
        if not self.is_for_vis:
            return {'image' : image, 'text' : text, 'text_mask' : text_mask}
        else: #need to return image_datum and text_datum separately, and also need a few more things
            with torch.no_grad():
                image_for_vis = image * self.unnorm_scale + self.unnorm_bias

            text_str_list = self.labels_dict[k]
            text_str_list.extend(['BLANK'] * (self.max_num_tokens - len(text_str_list)))
            image_datum = {'image' : image, 'image_for_vis' : image_for_vis, 'idx' : idx}
            text_datum = {'text' : text, 'text_mask' : text_mask, 'text_str_list' : text_str_list, 'caption_str' : self.caption_dict[k], 'idx' : idx}
            return image_datum, text_datum


    def __len__(self):
        return len(self.key_list)
