import os
import sys
import clip
from PIL import Image
import json
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.datasets import Flowers102, DTD, Food101
from eval_clip_ensemble_baseline_templates import TEMPLATES
from eval_clip_single_baseline_templates import SINGLE_TEMPLATE


class EvalDatasetBase(Dataset):

    def __init__(self, dataset_parent_dir, context_length, clip_model_type):
        self.dataset_parent_dir = dataset_parent_dir
        self.context_length = context_length
        self.clip_model_type = clip_model_type

    #override this if you don't have self.classnames
    def get_label_classnames(self, num_repetitions=1):
        return clip.tokenize([' '.join([classname] * num_repetitions) for classname in self.classnames], context_length=self.context_length)

    def get_single_templated_classnames(self):
        return clip.tokenize([self.single_template.format(classname) for classname in self.classnames], context_length=self.context_length)

    #override this if you don't have self.classnames and self.templates
    def get_templated_classnames(self):
        num_classes = len(self.classnames)
        num_templates = len(self.templates)
        texts = []
        for classname in self.classnames:
            for template in self.templates:
                texts.append(template.format(classname))

        texts = clip.tokenize(texts, context_length=self.context_length)
        texts = torch.unflatten(texts, dim=0, sizes=(num_classes, num_templates))
        return texts

    def get_image_bases(self, idxs):
        assert(False)

    def __getitem__(self, idx):
        assert(False)

    def __len__(self):
        assert(False)


class EvalDatasetFlowers102(EvalDatasetBase):

    def __init__(self, dataset_parent_dir, context_length, clip_model_type):
        super().__init__(dataset_parent_dir, context_length, clip_model_type)
        _, prepro = clip.load(self.clip_model_type, device='cpu')
        self.internal_dataset = Flowers102(os.path.join(self.dataset_parent_dir, 'Flowers102'), split='test', transform=prepro)
        with open(os.path.join(self.dataset_parent_dir, 'Flowers102', 'cat_to_name.json'), 'r') as f:
            cat_to_name_unprocessed = json.load(f)

        self.classnames = [cat_to_name_unprocessed[str(i+1)] for i in range(len(cat_to_name_unprocessed))]
        self.templates = TEMPLATES['Flowers102']
        self.single_template = SINGLE_TEMPLATE['Flowers102']

    def get_image_bases(self, idxs):
        return [os.path.basename(self.internal_dataset._image_files[idx]) for idx in idxs.numpy()]

    def __getitem__(self, idx):
        image, gt = self.internal_dataset[idx]
        return {'image' : image, 'gt' : torch.tensor(gt, dtype=torch.long), 'idx' : torch.tensor(idx, dtype=torch.long)}

    def __len__(self):
        return len(self.internal_dataset)


class EvalDatasetDTD(EvalDatasetBase):

    def __init__(self, dataset_parent_dir, context_length, clip_model_type):
        super().__init__(dataset_parent_dir, context_length, clip_model_type)
        _, prepro = clip.load(self.clip_model_type, device='cpu')
        self.internal_dataset = DTD(os.path.join(self.dataset_parent_dir, 'DTD'), split='test', partition=1, transform=prepro)
        with open(os.path.join(self.dataset_parent_dir, 'DTD', 'classnames.json'), 'r') as f:
            self.classnames = json.load(f)

        self.templates = TEMPLATES['DTD']
        self.single_template = SINGLE_TEMPLATE['DTD']

    def get_image_bases(self, idxs):
        return [os.path.basename(self.internal_dataset._image_files[idx]) for idx in idxs.numpy()]

    def __getitem__(self, idx):
        image, gt = self.internal_dataset[idx]
        return {'image' : image, 'gt' : torch.tensor(gt, dtype=torch.long), 'idx' : torch.tensor(idx, dtype=torch.long)}

    def __len__(self):
        return len(self.internal_dataset)


class EvalDatasetFood101(EvalDatasetBase):

    def __init__(self, dataset_parent_dir, context_length, clip_model_type):
        super().__init__(dataset_parent_dir, context_length, clip_model_type)
        _, prepro = clip.load(self.clip_model_type, device='cpu')
        self.internal_dataset = Food101(os.path.join(self.dataset_parent_dir, 'Food101'), split='test', transform=prepro)
        with open(os.path.join(self.dataset_parent_dir, 'Food101', 'classnames.json'), 'r') as f:
            self.classnames = json.load(f)

        self.templates = TEMPLATES['Food101']
        self.single_template = SINGLE_TEMPLATE['Food101']

    def get_image_bases(self, idxs):
        return [os.path.basename(self.internal_dataset._image_files[idx]) for idx in idxs.numpy()]

    def __getitem__(self, idx):
        image, gt = self.internal_dataset[idx]
        return {'image' : image, 'gt' : torch.tensor(gt, dtype=torch.long), 'idx' : torch.tensor(idx, dtype=torch.long)}

    def __len__(self):
        return len(self.internal_dataset)


class EvalDatasetUCF101(EvalDatasetBase):

    def __init__(self, dataset_parent_dir, context_length, clip_model_type):
        super().__init__(dataset_parent_dir, context_length, clip_model_type)
        _, self.prepro = clip.load(self.clip_model_type, device='cpu')
        with open(os.path.join(self.dataset_parent_dir, 'UCF101', 'split_zhou_UCF101.json'), 'r') as f:
            d = json.load(f)

        self.image_partialpath_list = []
        self.gt_list = []
        gt2classname = {}
        for image_partialpath, gt, classname_raw in d['test']:
            self.image_partialpath_list.append(image_partialpath)
            self.gt_list.append(gt)
            classname = classname_raw.replace('_', ' ')
            if gt in gt2classname:
                assert(classname == gt2classname[gt])

            gt2classname[gt] = classname

        assert(list(range(len(gt2classname))) == sorted(gt2classname.keys()))
        self.classnames = [gt2classname[gt] for gt in range(len(gt2classname))]
        self.templates = TEMPLATES['UCF101']
        self.single_template = SINGLE_TEMPLATE['UCF101']

    def get_image_bases(self, idxs):
        return [os.path.basename(self.image_partialpath_list[idx]) for idx in idxs.numpy()]

    def __getitem__(self, idx):
        image = self.prepro(Image.open(os.path.join(self.dataset_parent_dir, 'UCF101', 'UCF-101-midframes', self.image_partialpath_list[idx])))
        gt = self.gt_list[idx]
        return {'image' : image, 'gt' : torch.tensor(gt, dtype=torch.long), 'idx' : torch.tensor(idx, dtype=torch.long)}

    def __len__(self):
        return len(self.image_partialpath_list)


class EvalDatasetImageNet1K(EvalDatasetBase):

    def __init__(self, dataset_parent_dir, context_length, clip_model_type):
        super().__init__(dataset_parent_dir, context_length, clip_model_type)
        _, self.prepro = clip.load(self.clip_model_type, device='cpu')
        with open(os.path.join(self.dataset_parent_dir, 'ImageNet1K', 'dataset.pkl'), 'rb') as f:
            ds = pickle.load(f)

        self.image_partialpath_list = ds['image_partialpath_list']
        self.gt_list = ds['gt_list']
        self.classnames = ds['classnames']
        self.templates = TEMPLATES['ImageNet1K']
        self.single_template = SINGLE_TEMPLATE['ImageNet1K']

    def get_image_bases(self, idxs):
        return [os.path.basename(self.image_partialpath_list[idx]) for idx in idxs.numpy()]

    def __getitem__(self, idx):
        image = self.prepro(Image.open(os.path.join(self.dataset_parent_dir, 'ImageNet1K', 'ILSVRC2012_val', self.image_partialpath_list[idx])))
        gt = self.gt_list[idx]
        return {'image' : image, 'gt' : torch.tensor(gt, dtype=torch.long), 'idx' : torch.tensor(idx, dtype=torch.long)}

    def __len__(self):
        return len(self.image_partialpath_list)
