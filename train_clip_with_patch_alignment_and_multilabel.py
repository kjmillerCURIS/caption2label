import os
import sys
import numpy as np
import pickle
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from checkpoint_utils import get_model_optimizer_scheduler, save_checkpoint, is_at_fractional_checkpoint
from clip_training_utils import add_to_backbone_gradients_multilabel_patch_alignment, add_to_backbone_gradients_smallbatch_multilabel_patch_alignment
from image_text_dataset_multilabel import ImageTextDatasetMultilabel
from experiment_params.param_utils import get_params_key
from experiment_params.params import grab_params
from write_to_log_file import write_to_log_file


def genny_fn(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def get_data_genny(image_dir, text_filename_prefix, num_workers, params):
    p = params
    dataset = ImageTextDatasetMultilabel(image_dir, text_filename_prefix, p)
    dataloader = DataLoader(dataset, batch_size=p.batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    return genny_fn(dataloader), len(dataloader)


def train_clip_with_patch_alignment_and_multilabel(image_dir, text_filename_prefix, experiment_dir, num_workers, is_laclip=False):
    num_workers = int(num_workers)
    is_laclip = bool(int(is_laclip))

    write_to_log_file('getting params...')
    params_key = get_params_key(experiment_dir)
    p = grab_params(params_key)

    #check params
    assert(p.do_patch_alignment)
    assert(p.do_multilabel)

    #setup dataset and dataloader
    write_to_log_file('data...')
    data_genny, epoch_length = get_data_genny(image_dir, text_filename_prefix, num_workers, p)

    #telemetry
    telemetry = {'epoch_length' : epoch_length, 'train_losses' : []}
    telemetry_filename = os.path.join(experiment_dir, 'telemetry.pkl')
    if os.path.exists(telemetry_filename):
        with open(telemetry_filename, 'rb') as f:
            telemetry = pickle.load(f)

    #load model (initial or mid)
    write_to_log_file('loading/initializing model...')
    checkpoint_prefix = os.path.join(experiment_dir, 'checkpoints', 'checkpoint')
    os.makedirs(os.path.dirname(checkpoint_prefix), exist_ok=True)
    model,optimizer,scheduler,start_epoch,start_step_in_epoch,is_on_loaded_checkpoint = get_model_optimizer_scheduler(p, checkpoint_prefix, epoch_length, is_laclip=is_laclip)

    #model.train()
    for k in ['image_backbone_upper', 'image_embedder', 'text_backbone_upper', 'patch_temperature', 'clip_temperature']:
        if model[k] is not None:
            model[k].train()

    #and model.eval() for lowers
    for k in ['image_backbone_lower', 'text_backbone_lower']:
        if model[k] is not None:
            model[k].eval()

#    #DEBUG stuff
#    debug_step_counter = 0

    #epoch loop
    cur_start_step_in_epoch = start_step_in_epoch
    for epoch in tqdm(range(start_epoch, p.max_epochs)):
        
        write_to_log_file('start an epoch!')

        #whole save
        if not is_on_loaded_checkpoint:
            assert(cur_start_step_in_epoch == 0) #because we always initialize the model with step=0, right? and if this isn't the initial model, then we're on a fresh epoch which starts with step=0 
            save_checkpoint(model, optimizer, scheduler, epoch, 0, checkpoint_prefix, telemetry, telemetry_filename, is_final=False)

        #step loop
        for step_in_epoch in tqdm(range(cur_start_step_in_epoch, epoch_length)):

#            #DEBUG stuff
#            debug_checkpoint = {}
#            for k in ['image_backbone_upper', 'image_embedder', 'text_backbone_upper', 'patch_temperature', 'clip_temperature']:
#                if model[k] is not None:
#                    debug_checkpoint[k] = model[k].state_dict()
#
#            torch.save(debug_checkpoint,os.path.join(experiment_dir,'debug_checkpoints','debug_checkpoint-%09d.pth'%(debug_step_counter)))
#            if debug_step_counter % 10 == 0:
#                with open(telemetry_filename, 'wb') as f:
#                    pickle.dump(telemetry, f)
#
#            debug_step_counter += 1

            #fractional save
            if is_at_fractional_checkpoint(p, epoch, step_in_epoch, epoch_length) and not is_on_loaded_checkpoint:
                save_checkpoint(model, optimizer, scheduler, epoch, step_in_epoch, checkpoint_prefix, telemetry, telemetry_filename, is_final=False)

            is_on_loaded_checkpoint = False

            #get batch
            batch = next(data_genny)
            image_batch, text_batch, text_mask = batch['image'].cuda(), batch['text'].cuda(), batch['text_mask'].cuda()

            #optim zero
            for k in ['image_backbone_upper','image_embedder','text_backbone_upper','patch_temperature','clip_temperature']:
                if model[k] is not None:
                    optimizer[k].zero_grad()

            #backprop
            backprop_start_time = time.time()
            if p.oversize_batch_mode:
                loss = add_to_backbone_gradients_multilabel_patch_alignment(model['image_backbone_lower'], model['image_backbone_upper'], model['image_embedder'], model['text_backbone_lower'], model['text_backbone_upper'], model['patch_temperature'], model['clip_temperature'], image_batch, text_batch, text_mask, p.image_minibatch_size, p.text_minibatch_size, 1.0, p)
            else:
                loss = add_to_backbone_gradients_smallbatch_multilabel_patch_alignment(model['image_backbone_lower'], model['image_backbone_upper'], model['image_embedder'], model['text_backbone_lower'], model['text_backbone_upper'], model['patch_temperature'], model['clip_temperature'], image_batch, text_batch, text_mask, 1.0, p)

            backprop_end_time = time.time()

            #telemetry
            telemetry['train_losses'].append(loss.item())

#            #DEBUG STUFF
#            write_to_log_file('train loss is ' + str(loss.item()))

            #optim step
            for k in ['image_backbone_upper','image_embedder','text_backbone_upper','patch_temperature','clip_temperature']:
                if model[k] is not None:
                    optimizer[k].step()

            #scheduler step
            for k in ['image_backbone_upper','image_embedder','text_backbone_upper','patch_temperature','clip_temperature']:
                if model[k] is not None:
                    scheduler[k].step()

        #reset for next epoch
        cur_start_step_in_epoch = 0

    #final save
    save_checkpoint(model, optimizer, scheduler, epoch, step_in_epoch, checkpoint_prefix, telemetry, telemetry_filename, is_final=True)


def usage():
    print('Usage: python train_clip_with_patch_alignment_and_multilabel.py <image_dir> <text_filename_prefix> <experiment_dir> <num_workers> [<is_laclip>=False]')


if __name__ == '__main__':
    train_clip_with_patch_alignment_and_multilabel(*(sys.argv[1:]))
