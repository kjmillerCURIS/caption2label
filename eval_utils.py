import os
import sys
import clip
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from clip_training_utils import special_forward, compute_individual_label_similarities, input_minibatcher
from checkpoint_utils import load_patch_alignment_model_from_checkpoint
from clip_backbone_utils import load_either_clip_startpoint
from eval_dataset import EvalDatasetFlowers102, EvalDatasetDTD, EvalDatasetFood101, EvalDatasetUCF101, EvalDatasetImageNet1K
from class_prevalence_utils import get_class_mask


#will map string to name of dataset class (in the Python sense)
DATASET_DICT = {'Flowers102' : EvalDatasetFlowers102, 'DTD' : EvalDatasetDTD, 'Food101' : EvalDatasetFood101, 'UCF101' : EvalDatasetUCF101, 'ImageNet1K' : EvalDatasetImageNet1K}

#just use the full context length, we're not finetuning the positional embeddings (yet) so you really don't have to worry about any discontinuities due to the short context length used during training
INFERENCE_CONTEXT_LENGTH = 77 #this is just the default

IMAGE_BATCH_SIZE = 128
TEXT_MINIBATCH_SIZE = 128
NUM_WORKERS = 2
BASELINE_CLIP_MODEL_TYPE = 'ViT-B/16'


def set_eval_model(model):
    for k in sorted(model.keys()):
        if model[k] is not None:
            model[k].eval()


#texts should be a tensor containing the tokenized classnames, i.e. num_classes x context_length
#will return tensor with shape num_classes x embedding_size
def embed_classnames_for_patch_alignment_and_multilabel(texts, model, params):
    if torch.is_grad_enabled():
        with torch.no_grad():
            embed_classnames_for_patch_alignment_and_multilabel(texts, model, params)

    set_eval_model(model)

    p = params
    text_embeddings = []
    for text_minibatch, _, __ in input_minibatcher(texts, TEXT_MINIBATCH_SIZE):
        if model['text_backbone_lower'] is not None:
            text_minibatch = model['text_backbone_lower'](text_minibatch)

        text_embeddings.append(special_forward(model['text_backbone_lower'], [model['text_backbone_upper']], text_minibatch))

    text_embeddings = torch.cat(text_embeddings, dim=0)
    return text_embeddings


#text_embeddings should have shape num_classes x embedding_size
#will give back predictions as ints
#everything starts and ends as tensors on the GPU
def predict_batch_with_patch_alignment_and_multilabel(image_batch, text_embeddings, model, params, class_mask=None):
    if torch.is_grad_enabled():
        with torch.no_grad():
            predict_batch_with_patch_alignment_and_multilabel(image_batch, text_embeddings, model, params, class_mask=class_mask)

    set_eval_model(model)

    p = params
    batch_size_i = image_batch.shape[0]
    num_classes = text_embeddings.shape[0]
    if model['image_backbone_lower'] is not None:
        image_batch = model['image_backbone_lower'](image_batch)

    image_patch_embeddings = special_forward(model['image_backbone_lower'], [model['image_backbone_upper'], model['image_embedder']], image_batch)
    text_embeddings = torch.unsqueeze(text_embeddings, dim=1) #num_classes x 1 x embedding_size
    cossims = compute_individual_label_similarities(text_embeddings, image_patch_embeddings, model['patch_temperature'], p, allow_different_batch_sizes=True)
    assert(cossims.shape == (batch_size_i, num_classes, 1))
    cossims = torch.squeeze(cossims, dim=-1)
    assert(cossims.shape == (batch_size_i, num_classes))
    if class_mask is not None: #set the cossims of invalid classes to -inf so they won't be chosen
        assert(np.any(class_mask))
        cossims[:,~class_mask] = -np.inf

    preds = torch.argmax(cossims, dim=1)
    if class_mask is not None:
        assert(np.all(class_mask[preds.detach().cpu().numpy()]))

    return preds


#texts should be a tensor containing the tokenized templated classnames, i.e. num_classes x num_templates x context_length
#will return tensor with shape num_classes x embedding_size
def embed_templated_classnames_for_clip_ensemble_baseline(texts, clip_model):
    if torch.is_grad_enabled():
        with torch.no_grad():
            embed_templated_classnames_for_clip_ensemble_baseline(texts, clip_model)

    clip_model.eval()
    num_classes = texts.shape[0]
    num_templates = texts.shape[1]
    flattened_texts = torch.flatten(texts, start_dim=0, end_dim=1)

    flattened_text_embeddings = []
    for flattened_text_minibatch, _, __ in input_minibatcher(flattened_texts, TEXT_MINIBATCH_SIZE):
        flattened_text_embeddings.append(clip_model.encode_text(flattened_text_minibatch))

    flattened_text_embeddings = torch.cat(flattened_text_embeddings, dim=0)
    text_embeddings = torch.unflatten(flattened_text_embeddings, dim=0, sizes=(num_classes, num_templates))
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    text_embeddings = torch.mean(text_embeddings, dim=1, keepdim=False)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    return text_embeddings


#texts should have dimensions num_classes x context_length
#will return tensor with shape num_classes x embedding_size
def embed_1D_texts_clip_baseline(texts, clip_model):
    if torch.is_grad_enabled():
        with torch.no_grad():
            embed_1D_texts_clip_baseline(texts, clip_model)

    clip_model.eval()
    num_classes = texts.shape[0]

    text_embeddings = []
    for text_minibatch, _, __ in input_minibatcher(texts, TEXT_MINIBATCH_SIZE):
        text_embeddings.append(clip_model.encode_text(text_minibatch))

    text_embeddings = torch.cat(text_embeddings, dim=0)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    return text_embeddings


#text_embeddings should be shape num_classes x embedding_size
def predict_batch_any_clip_baseline(image_batch, text_embeddings, clip_model, class_mask=None):
    if torch.is_grad_enabled():
        with torch.no_grad():
            predict_batch_clip_baseline(image_batch, text_embeddings, clip_model, class_mask=class_mask)

    clip_model.eval()
    image_embeddings = clip_model.encode_image(image_batch)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

    cossims = image_embeddings @ text_embeddings.t()
    if class_mask is not None: #set the cossims of invalid classes to -inf so they won't be chosen
        assert(np.any(class_mask))
        cossims[:,~class_mask] = -np.inf

    preds = torch.argmax(cossims, dim=1)
    if class_mask is not None:
        assert(np.all(class_mask[preds.detach().cpu().numpy()]))

    return preds


def compute_accuracy(results, class_mask=None):
    total_counter = {}
    correct_counter = {}
    total = 0
    correct = 0
    for image_base in sorted(results['gt_dict'].keys()):
        total += 1
        gt = results['gt_dict'][image_base]
        pred = results['pred_dict'][image_base]
        if class_mask is not None:
            assert(class_mask[gt])
            assert(class_mask[pred])

        if gt not in total_counter:
            total_counter[gt] = 0
            correct_counter[gt] = 0

        total_counter[gt] += 1
        if pred == gt:
            correct += 1
            correct_counter[gt] += 1

    results['unbalanced_accuracy_as_percentage'] = 100.0 * correct / total
    results['balanced_accuracy_as_percentage'] = 100.0 * np.mean([correct_counter[gt] / total_counter[gt] for gt in sorted(total_counter.keys())])
    return results


#returns None if none of the classes get through, otherwise returns a new batch that only has examples with valid gt classes
#this does NOT reindex the classes!
def filter_batch_by_class(batch, class_mask):
    with torch.no_grad():
        gts = batch['gt'].detach().numpy()
        is_valid = class_mask[gts]
        if not np.any(is_valid):
            return None

        new_batch = {}
        for k in sorted(batch.keys()):
            new_batch[k] = batch[k].detach()[is_valid]

        return new_batch


#specify class_prevalence_dict_filename to do evaluation on ONLY the prevalent classes, else keep it as None to evaluate on ALL classes
def evaluate_patch_alignment_with_multilabel(dataset_name, dataset_parent_dir, checkpoint, params, class_prevalence_dict_filename=None, is_laclip=False):
    p = params

    class_mask = None
    if class_prevalence_dict_filename is not None:
        class_mask = get_class_mask(class_prevalence_dict_filename, dataset_name)

    dataset = DATASET_DICT[dataset_name](dataset_parent_dir, INFERENCE_CONTEXT_LENGTH, p.clip_model_type)
    texts = dataset.get_label_classnames().cuda()
    results = {'gt_dict' : {}, 'pred_dict' : {}}
    dataloader = DataLoader(dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    model = load_patch_alignment_model_from_checkpoint(p, checkpoint, is_laclip=is_laclip)
    if model['text_backbone_lower'] is not None:
        model['text_backbone_lower'].change_context_length(INFERENCE_CONTEXT_LENGTH)

    if model['text_backbone_upper'] is not None:
        model['text_backbone_upper'].change_context_length(INFERENCE_CONTEXT_LENGTH)

    set_eval_model(model)
    with torch.no_grad():
        text_embeddings = embed_classnames_for_patch_alignment_and_multilabel(texts, model, p)

    for the_batch in tqdm(dataloader):
        if class_mask is not None:
            batch = filter_batch_by_class(the_batch, class_mask)
            if batch is None:
                continue

        else:
            batch = the_batch

        image_batch = batch['image'].cuda()
        image_bases = dataset.get_image_bases(batch['idx'])
        gts = batch['gt'].numpy()
        with torch.no_grad():
            preds = predict_batch_with_patch_alignment_and_multilabel(image_batch, text_embeddings, model, p, class_mask=class_mask)
            preds = preds.cpu().numpy()

        for image_base, pred, gt in zip(image_bases, preds, gts):
            results['pred_dict'][image_base] = pred
            results['gt_dict'][image_base] = gt

    results = compute_accuracy(results, class_mask=class_mask)
    return results


def evaluate_clip_ensemble_baseline(dataset_name, dataset_parent_dir, class_prevalence_dict_filename=None, is_laclip=False):
    class_mask = None
    if class_prevalence_dict_filename is not None:
        class_mask = get_class_mask(class_prevalence_dict_filename, dataset_name)

    dataset = DATASET_DICT[dataset_name](dataset_parent_dir, INFERENCE_CONTEXT_LENGTH, BASELINE_CLIP_MODEL_TYPE)
    texts = dataset.get_templated_classnames().cuda()
    results = {'gt_dict' : {}, 'pred_dict' : {}}
    dataloader = DataLoader(dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    clip_model = load_either_clip_startpoint(BASELINE_CLIP_MODEL_TYPE, is_laclip=is_laclip)
    clip_model.float()
    clip_model.eval()
    with torch.no_grad():
        text_embeddings = embed_templated_classnames_for_clip_ensemble_baseline(texts, clip_model)

    for the_batch in tqdm(dataloader):
        if class_mask is not None:
            batch = filter_batch_by_class(the_batch, class_mask)
            if batch is None:
                continue
        else:
            batch = the_batch

        image_batch = batch['image'].cuda()
        image_bases = dataset.get_image_bases(batch['idx'])
        gts = batch['gt'].numpy()
        with torch.no_grad():
            preds = predict_batch_any_clip_baseline(image_batch, text_embeddings, clip_model, class_mask=class_mask)
            preds = preds.cpu().numpy()

        for image_base, pred, gt in zip(image_bases, preds, gts):
            results['pred_dict'][image_base] = pred
            results['gt_dict'][image_base] = gt

    results = compute_accuracy(results, class_mask=class_mask)
    return results


def evaluate_clip_single_template_baseline(dataset_name, dataset_parent_dir, class_prevalence_dict_filename=None, is_laclip=False):
    class_mask = None
    if class_prevalence_dict_filename is not None:
        class_mask = get_class_mask(class_prevalence_dict_filename, dataset_name)

    dataset = DATASET_DICT[dataset_name](dataset_parent_dir, INFERENCE_CONTEXT_LENGTH, BASELINE_CLIP_MODEL_TYPE)
    texts = dataset.get_single_templated_classnames().cuda()
    results = {'gt_dict' : {}, 'pred_dict' : {}}
    dataloader = DataLoader(dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    clip_model = load_either_clip_model(BASELINE_CLIP_MODEL_TYPE, is_laclip=is_laclip)
    clip_model.float()
    clip_model.eval()
    with torch.no_grad():
        text_embeddings = embed_1D_texts_clip_baseline(texts, clip_model)

    for the_batch in tqdm(dataloader):
        if class_mask is not None:
            batch = filter_batch_by_class(the_batch, class_mask)
            if batch is None:
                continue
        else:
            batch = the_batch

        image_batch = batch['image'].cuda()
        image_bases = dataset.get_image_bases(batch['idx'])
        gts = batch['gt'].numpy()
        with torch.no_grad():
            preds = predict_batch_any_clip_baseline(image_batch, text_embeddings, clip_model, class_mask=class_mask)
            preds = preds.cpu().numpy()

        for image_base, pred, gt in zip(image_bases, preds, gts):
            results['pred_dict'][image_base] = pred
            results['gt_dict'][image_base] = gt

    results = compute_accuracy(results, class_mask=class_mask)
    return results


def evaluate_clip_repeated_classname_baseline(dataset_name, dataset_parent_dir, num_repetitions, class_prevalence_dict_filename=None, is_laclip=False):
    class_mask = None
    if class_prevalence_dict_filename is not None:
        class_mask = get_class_mask(class_prevalence_dict_filename, dataset_name)

    dataset = DATASET_DICT[dataset_name](dataset_parent_dir, INFERENCE_CONTEXT_LENGTH, BASELINE_CLIP_MODEL_TYPE)
    texts = dataset.get_label_classnames(num_repetitions=num_repetitions).cuda()
    results = {'gt_dict' : {}, 'pred_dict' : {}}
    dataloader = DataLoader(dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    clip_model = load_either_clip_model(BASELINE_CLIP_MODEL_TYPE, is_laclip=is_laclip)
    clip_model.float()
    clip_model.eval()
    with torch.no_grad():
        text_embeddings = embed_1D_texts_clip_baseline(texts, clip_model)

    for the_batch in tqdm(dataloader):
        if class_mask is not None:
            batch = filter_batch_by_class(the_batch, class_mask)
            if batch is None:
                continue
        else:
            batch = the_batch

        image_batch = batch['image'].cuda()
        image_bases = dataset.get_image_bases(batch['idx'])
        gts = batch['gt'].numpy()
        with torch.no_grad():
            preds = predict_batch_any_clip_baseline(image_batch, text_embeddings, clip_model, class_mask=class_mask)
            preds = preds.cpu().numpy()

        for image_base, pred, gt in zip(image_bases, preds, gts):
            results['pred_dict'][image_base] = pred
            results['gt_dict'][image_base] = gt

    results = compute_accuracy(results, class_mask=class_mask)
    return results
