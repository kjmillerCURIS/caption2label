import os
import sys
import glob
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import math
import numpy as np
import pickle
import torch
from torch import optim
from clip_backbone_utils import get_backbones_with_patch_alignment


#model will be dict that contains image_backbone_lower, image_backbone_upper, image_embedder, text_backbone_lower, text_backbone_upper, patch_temperature, clip_temperature
#same deal for optimizer and scheduler, except they don't contain any lowers
#also, the lowers won't actually be included in the checkpoint, they'll just come in through the OpenAI code


def is_at_fractional_checkpoint(params, epoch, step_in_epoch, epoch_length):
    p = params
    for fraction in p.fractional_checkpoints:
        target_epoch = int(math.floor(fraction))
        target_step_in_epoch = int(round((fraction - target_epoch) * epoch_length))
        if epoch == target_epoch and step_in_epoch == target_step_in_epoch:
            return True

    return False


def get_checkpoint_epoch_step(checkpoint_filename):
    ss = os.path.splitext(os.path.basename(checkpoint_filename))[0].split('-')[-2:]
    return (int(ss[0]), int(ss[1]))


def load_patch_alignment_model_from_checkpoint(params, checkpoint, is_laclip=False):
    p = params
    image_backbone_lower, image_backbone_upper, image_embedder, text_backbone_lower, text_backbone_upper, patch_temperature, clip_temperature = get_backbones_with_patch_alignment(p, is_laclip=is_laclip)
    model = {}
    model['image_backbone_lower'] = image_backbone_lower
    model['image_backbone_upper'] = image_backbone_upper
    model['image_embedder'] = image_embedder
    model['text_backbone_lower'] = text_backbone_lower
    model['text_backbone_upper'] = text_backbone_upper
    model['patch_temperature'] = patch_temperature
    model['clip_temperature'] = clip_temperature
    for k in ['image_backbone_upper', 'image_embedder', 'text_backbone_upper', 'patch_temperature', 'clip_temperature']:
        if model[k] is None:
            continue

        model[k].load_state_dict(checkpoint['model_state_dict'][k])

    return model


#load a checkpoint (or get the initial model)
def get_model_optimizer_scheduler(params, checkpoint_prefix, epoch_length, is_laclip=False):
    p = params
    assert(not os.path.exists(checkpoint_prefix + '-FINAL.pth'))

    #create model
    image_backbone_lower, image_backbone_upper, image_embedder, text_backbone_lower, text_backbone_upper, patch_temperature, clip_temperature = get_backbones_with_patch_alignment(p, is_laclip=is_laclip)
    model = {}
    model['image_backbone_lower'] = image_backbone_lower
    model['image_backbone_upper'] = image_backbone_upper
    model['image_embedder'] = image_embedder
    model['text_backbone_lower'] = text_backbone_lower
    model['text_backbone_upper'] = text_backbone_upper
    model['patch_temperature'] = patch_temperature
    model['clip_temperature'] = clip_temperature

    #create optimizer
    optimizer = {}
    exclude = lambda na, pa: pa.ndim < 2 or 'bn' in na or 'ln' in na or 'norm' in na or 'bias' in na or 'logit_scale' in na
    include = lambda na, pa: not exclude(na, pa)
    for k in ['image_backbone_upper','image_embedder','text_backbone_upper','patch_temperature','clip_temperature']:
        if model[k] is None:
            continue

        named_parameters = list(model[k].named_parameters())
        gain_or_bias_params = [pa for na, pa in named_parameters if exclude(na, pa) and pa.requires_grad]
        rest_params = [pa for na, pa in named_parameters if include(na, pa) and pa.requires_grad]
        if p.optimizer_type[k] == 'AdamW':
            optimizer[k] = optim.AdamW(
                [
                    {'params': gain_or_bias_params, 'weight_decay': 0.0},
                    {'params': rest_params, 'weight_decay': p.weight_decay[k]},
                ],
                lr=p.learning_rate[k],
                betas=(p.beta1[k], p.beta2[k]),
                eps=p.epsilon[k],
            )
        else:
            assert(False)

    #create scheduler (in this case, everyone gets a scheduler, there's no "None" here, that was only for disentanglement model)
    scheduler = {}
    for k in ['image_backbone_upper','image_embedder','text_backbone_upper','patch_temperature','clip_temperature']:
        if model[k] is None:
            continue

        if p.scheduler_type[k] == 'LinearWarmupCosineAnnealingLR':
            scheduler[k] = LinearWarmupCosineAnnealingLR(optimizer[k], int(round(p.warmup_epochs[k] * epoch_length)), int(round(p.max_epochs * epoch_length)))
        else:
            assert(False)

    #find latest checkpoint and update states from it if it exists
    checkpoint_filenames = sorted(glob.glob(checkpoint_prefix + '-*-*.pth'))
    checkpoint_epoch_step_list = [get_checkpoint_epoch_step(checkpoint_filename) for checkpoint_filename in checkpoint_filenames]
    if len(checkpoint_epoch_step_list) == 0: #if this is the initial model
        return model, optimizer, scheduler, 0, 0, False
    else:
        epoch_step_filename_list = [(epoch, step, filename) for (epoch, step), filename in zip(checkpoint_epoch_step_list, checkpoint_filenames)]
        _, __, best_filename = sorted(epoch_step_filename_list, reverse=True)[0]
        checkpoint = torch.load(best_filename)
        for k in ['image_backbone_upper', 'image_embedder', 'text_backbone_upper', 'patch_temperature', 'clip_temperature']:
            if model[k] is None:
                continue

            model[k].load_state_dict(checkpoint['model_state_dict'][k])
            optimizer[k].load_state_dict(checkpoint['optimizer_state_dict'][k])
            scheduler[k].load_state_dict(checkpoint['scheduler_state_dict'][k])

        epoch = checkpoint['epoch']
        step_within_epoch = checkpoint['step_within_epoch']
        return model, optimizer, scheduler, epoch, step_within_epoch, True


#save a checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, step_within_epoch, checkpoint_prefix, telemetry, telemetry_filename, is_final=False):
    checkpoint = {'epoch':epoch, 'step_within_epoch':step_within_epoch, 'model_state_dict':{}, 'optimizer_state_dict':{}, 'scheduler_state_dict':{}}
    for k in ['image_backbone_upper', 'image_embedder', 'text_backbone_upper', 'patch_temperature', 'clip_temperature']:
        if model[k] is None:
            continue

        checkpoint['model_state_dict'][k] = model[k].state_dict()
        checkpoint['optimizer_state_dict'][k] = optimizer[k].state_dict()
        checkpoint['scheduler_state_dict'][k] = scheduler[k].state_dict()

    if not is_final:
        torch.save(checkpoint, checkpoint_prefix + '-%03d-%09d.pth'%(epoch, step_within_epoch))
    else:
        torch.save(checkpoint, checkpoint_prefix + '-FINAL.pth')

    with open(telemetry_filename, 'wb') as f:
        pickle.dump(telemetry, f)
