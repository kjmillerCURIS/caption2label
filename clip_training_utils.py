import os
import sys
import clip
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from write_to_log_file import write_to_log_file


#assume that text_embeddings and image_patch_embeddings are ALREADY NORMALIZED AND FP16
#no need for masking out the empty labels, that'll happen later
#I'll probably reuse this for PACL by just adding in an extra dimension for max_num_labels
def compute_individual_label_similarities(text_embeddings,image_patch_embeddings,patch_temperature,params,allow_different_batch_sizes=False,return_extras=False):
    p = params

    batch_size_i = image_patch_embeddings.shape[0]
    batch_size_t = text_embeddings.shape[0]
    if not allow_different_batch_sizes:
        assert(batch_size_i == batch_size_t)

    max_num_labels = text_embeddings.shape[1]
    num_patches = image_patch_embeddings.shape[1]
    embedding_size = text_embeddings.shape[2]
    assert(image_patch_embeddings.shape[2] == embedding_size)

    #drop CLF token if we're supposed to do that
    if p.drop_clf_token:
        image_patch_embeddings = image_patch_embeddings[:,1:,:]
        num_patches = image_patch_embeddings.shape[1]

    #first we need patch-level similarities...
    #shape of these is (batch_size_i, num_patches, batch_size_t, max_num_labels)
    #start by partially flattening the inputs, then matmul, then unflatten the output
    xi = torch.flatten(image_patch_embeddings, start_dim=0, end_dim=1)
    xt = torch.flatten(text_embeddings, start_dim=0, end_dim=1)
    mprod = xi @ xt.t() #(batch_size_i * num_patches, batch_size_t * max_num_labels)
    assert(mprod.shape == (batch_size_i * num_patches, batch_size_t * max_num_labels))
    patch_cossims = torch.unflatten(torch.unflatten(mprod, dim=-1, sizes=(batch_size_t, max_num_labels)), dim=0, sizes=(batch_size_i, num_patches))
    assert(patch_cossims.shape == (batch_size_i, num_patches, batch_size_t, max_num_labels))

    #now the softmax...
    patch_probs = F.softmax(patch_temperature(patch_cossims), dim=1)
#    write_to_log_file('all patch_probs sum up to at least ' + str(torch.min(torch.sum(patch_probs, dim=1))) + ' and at most ' + str(torch.max(torch.sum(patch_probs, dim=1))))

    #now the weighted average...
    #flatten the weights, permute, bmm, and unflatten
    #final shape will be (batch_size_i, batch_size_t, max_num_labels, embedding_size)
    xp = torch.flatten(patch_probs, start_dim=2, end_dim=3)
    xp = torch.permute(xp, (0, 2, 1))
    assert(xp.shape == (batch_size_i, batch_size_t * max_num_labels, num_patches))
    bprod = torch.bmm(xp, image_patch_embeddings)
    assert(bprod.shape == (batch_size_i, batch_size_t * max_num_labels, embedding_size))
    image_embeddings = torch.unflatten(bprod, dim=1, sizes=(batch_size_t, max_num_labels))
    assert(image_embeddings.shape == (batch_size_i, batch_size_t, max_num_labels, embedding_size))

    #renormalize...
#    write_to_log_file('blended image embedding minimum norm is ' + str(torch.min(image_embeddings.norm(dim=-1, keepdim=True))))
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

    #image-level similarities...
    #just have to broadcast and multiply text_embeddings onto image_embeddings, then sum away the last dim
    cossims = torch.sum(image_embeddings * torch.unsqueeze(text_embeddings, dim=0), dim=-1, keepdim=False)
    assert(cossims.shape == (batch_size_i, batch_size_t, max_num_labels))

    if return_extras:
        return cossims, {'patch_cossims' : patch_cossims, 'patch_probs' : patch_probs}
    else:
        return cossims


#note: this will modify cossims in-place
def aggregate_similarities(cossims, text_mask, params, return_extras=False):
    p = params
    assert(p.do_patch_alignment)
    assert(p.do_multilabel)

    batch_size_i = cossims.shape[0]
    batch_size_t = cossims.shape[1]
    assert(batch_size_i == batch_size_t)
    max_num_labels = cossims.shape[2]
    assert(text_mask.shape == (batch_size_t, max_num_labels))

    cossims[:, text_mask == 0] = np.nan

    if p.aggregation_type == 'avg':
        out = torch.nanmean(cossims, dim=-1, keepdim=False)
        if return_extras:
            is_included = torch.unsqueeze(text_mask > 0, 0)

    elif p.aggregation_type == 'avg_drop_percentile':
        thresholds = torch.nanquantile(cossims.detach().to(torch.float32), p.avg_drop_percentile_val, dim=-1, keepdim=True).to(torch.float16) #keepdim so we can broadcast
        cossims[cossims < thresholds] = np.nan
        out = torch.nanmean(cossims, dim=-1, keepdim=False)
        if return_extras:
            is_included = (~torch.isnan(cossims)) & torch.unsqueeze(text_mask > 0, 0)
    else:
        assert(False)

    if return_extras:
        return out, {'is_included' : is_included}
    else:
        return out


def vanilla_CLIP_loss_from_cossims(cossims, clip_temperature):
    assert(len(cossims.shape) == 2)
    assert(cossims.shape[0] == cossims.shape[1])
    logits_per_image = clip_temperature(cossims)
    logits_per_text = logits_per_image.t()
    labels = torch.arange(cossims.shape[0], device=cossims.device)
    loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2.0
    return loss


#for multilabel loss: should aggregate similarities for each label before taking softmax. Start with average as aggregation method, but ideally we'd also allow something like dropping the lower half, lower 10%, median filtering, etc.
#assume that text_embeddings and image_patch_embeddings are ALREADY NORMALIZED AND FP16
#text_embeddings should have shape (batch_size, max_num_labels, embedding_size)
#text_mask should be binary and have shape (batch_size, max_num_labels)
#image_patch_embeddings should have shape (batch_size, num_patches, embedding_size)
#will return a single number, which you can backprop down to text_embeddings and image_patch_embeddings
#Note that there are NO LEARNABLE THINGS in this function, except the softmax temperatures. E.g. we assume that the custom image embedder already did its thing and now we're working with the outputs of that
#we don't assume that anything's been normalize though...
#oh, and yes, we assume that image_patch_embeddings already has the CLS token appended to it, so num_patches should account for that
#btw, the "temperatures" are actually Incubators (see clip_backbone_utils.py) containing the negative logs of the temperatures
def compute_loss_from_embeddings(text_embeddings, text_mask, image_patch_embeddings, patch_temperature, clip_temperature, params):
    p = params
    assert(p.do_patch_alignment)
    assert(p.do_multilabel)

    batch_size_i = image_patch_embeddings.shape[0]
    batch_size_t = text_embeddings.shape[0]
    assert(batch_size_i == batch_size_t)
    max_num_labels = text_embeddings.shape[1]
    num_patches = image_patch_embeddings.shape[1]
    embedding_size = text_embeddings.shape[2]
    assert(image_patch_embeddings.shape[2] == embedding_size)
    assert(max_num_labels == p.max_num_labels)

    cossims = compute_individual_label_similarities(text_embeddings,image_patch_embeddings,patch_temperature,params)
    assert(cossims.shape == (batch_size_i, batch_size_t, max_num_labels))

    cossims = aggregate_similarities(cossims, text_mask, params)
    assert(cossims.shape == (batch_size_i, batch_size_t))

    loss = vanilla_CLIP_loss_from_cossims(cossims, clip_temperature)

    return loss


def input_minibatcher(input_batch, minibatch_size):
    if isinstance(input_batch, tuple):
        assert(len(input_batch) == 2)
        N = input_batch[0].shape[0]
    else:
        N = input_batch.shape[0]

    chunk_start = 0
    while chunk_start < N:
        chunk_end = min(chunk_start + minibatch_size, N)
        if isinstance(input_batch, tuple):
            yield (input_batch[0][chunk_start:chunk_end], input_batch[1][chunk_start:chunk_end]), chunk_start, chunk_end
        else:
            yield input_batch[chunk_start:chunk_end], chunk_start, chunk_end

        chunk_start = chunk_end


#this is responsible for normalizing the embeddings AND converting them to fp16 (to save memory for top part of the computation, which shouldn't be too precision-sensitive)
def special_forward(lower, uppers, X, print_minimum_norm=False):
#    if lower is not None:
#        with torch.no_grad():
#            X = lower(X)

    for upper in uppers:
        if upper is not None:
            X = upper(X)

    if print_minimum_norm:
        write_to_log_file('image patch minimum norm is ' + str(torch.min(X.norm(dim=-1, keepdim=True))))

    X = X / X.norm(dim=-1, keepdim=True)
    X = X.to(torch.float16)

    return X


def gather_image_patch_embeddings_nograd(image_backbone_lower, image_backbone_upper, image_embedder, image_batch, image_minibatch_size):
    with torch.no_grad():
        if image_backbone_upper is not None:
            image_backbone_upper.eval()

        image_embedder.eval()
        embeddings = []
        for image_minibatch, _, __ in input_minibatcher(image_batch, image_minibatch_size):
            embeddings.append(special_forward(image_backbone_lower, [image_backbone_upper, image_embedder], image_minibatch))

        embeddings = torch.cat(embeddings, dim=0)

    return embeddings


def gather_text_embeddings_nograd(text_backbone_lower, text_backbone_upper, text_batch, text_minibatch_size):
    with torch.no_grad():
        if text_backbone_upper is not None:
            text_backbone_upper.eval()

        embeddings = []
        for text_minibatch, _, __ in input_minibatcher(text_batch, text_minibatch_size):
            embeddings.append(special_forward(text_backbone_lower, [text_backbone_upper], text_minibatch))

        embeddings = torch.cat(embeddings, dim=0)

    return embeddings


def accumulate_into_image_backbones(image_backbone_lower, image_backbone_upper, image_embedder, image_batch, image_minibatch_size, image_top_grad, loss_weight):
    if image_backbone_upper is not None:
        image_backbone_upper.train()
    
    image_embedder.train()
    for image_minibatch, start_index, end_index in input_minibatcher(image_batch, image_minibatch_size):
        image_patch_embeddings = special_forward(image_backbone_lower, [image_backbone_upper, image_embedder], image_minibatch)
        partial_loss = loss_weight * torch.sum(image_patch_embeddings * image_top_grad[start_index:end_index])
        partial_loss.backward()


def accumulate_into_text_backbones(text_backbone_lower, text_backbone_upper, text_batch, text_minibatch_size, text_top_grad, loss_weight):
    if text_backbone_upper is not None:
        text_backbone_upper.train()
    
    for text_minibatch, start_index, end_index in input_minibatcher(text_batch, text_minibatch_size):
        text_embeddings = special_forward(text_backbone_lower, [text_backbone_upper], text_minibatch)
        partial_loss = loss_weight * torch.sum(text_embeddings * text_top_grad[start_index:end_index])
        partial_loss.backward()


def get_training_mode(image_backbone_upper, image_embedder, text_backbone_upper):
    names = ['image_backbone_upper', 'image_embedder', 'text_backbone_upper']
    backbones = [image_backbone_upper,image_embedder,text_backbone_upper]
    is_training_dict = {}
    for name, backbone in zip(names, backbones):
        if backbone is None:
            continue

        is_training_dict[name] = backbone.training

    return is_training_dict


def set_training_mode(image_backbone_upper,image_embedder,text_backbone_upper,is_training_dict):
    names = ['image_backbone_upper', 'image_embedder', 'text_backbone_upper']
    backbones = [image_backbone_upper,image_embedder,text_backbone_upper]
    for name, backbone in zip(names, backbones):
        if backbone is None:
            continue

        if is_training_dict[name]:
            backbone.train()
        else:
            backbone.eval()


#image_backbone_lower, text_backbone_lower are frozen. None means identity
#image_backbone_upper, text_backbone_upper are learnable. None means identity
#image_backbone_upper o image_backbone_lower should give (*, num_patches, pre_embedding_size)
#text_backbone_upper o text_backbone_lower should give (*, embedding_size) (we'll flatten the multilabels to accomodate this)
#image_embedder will always be learnable and not None
#it will turn (*, num_patches, pre_embedding_size) into (*, num_patches, embedding_size), internally flattening and unflattening if necesssary
#patch_temperature, clip_temperature are learnable scalars
#image_batch should be shape (*, C, H, W), as always, preprocessing/augmentations already done, etc.
#text_batch should be shape (*, max_num_labels, max_num_tokens) and be integers
#text_mask should be shape (*, max_num_labels) and be binary
def add_to_backbone_gradients_multilabel_patch_alignment(image_backbone_lower, image_backbone_upper, image_embedder, text_backbone_lower, text_backbone_upper, patch_temperature, clip_temperature, image_batch, text_batch, text_mask, image_minibatch_size, text_minibatch_size, loss_weight, params):
    p = params
    assert(p.do_patch_alignment)
    assert(p.do_multilabel)

    batch_size_i = image_batch.shape[0]
    batch_size_t = text_batch.shape[0]
    assert(batch_size_i == batch_size_t)
    max_num_labels = text_batch.shape[1]
    assert(text_mask.shape == (batch_size_t, max_num_labels))

    #get training mode
    is_training_dict = get_training_mode(image_backbone_upper,image_embedder,text_backbone_upper)

    #precompute the lower outputs
    if image_backbone_lower is not None:
        with torch.no_grad():
            image_batch = image_backbone_lower(image_batch)

    if text_backbone_lower is not None:
        with torch.no_grad():
            text_batch = text_backbone_lower(text_batch)

    #gather image and text embeddings (without any autograd) (gather_embeddings_nograd)
    image_patch_embeddings = gather_image_patch_embeddings_nograd(image_backbone_lower, image_backbone_upper, image_embedder, image_batch, image_minibatch_size)
    text_embeddings = gather_text_embeddings_nograd(text_backbone_lower, text_backbone_upper, text_batch, text_minibatch_size)

    #get gradient of loss w.r.t. embeddings (compute_CLIP_grad_wrt_embeddings)
    image_patch_embeddings.requires_grad_()
    text_embeddings.requires_grad_()
    loss = compute_loss_from_embeddings(text_embeddings, text_mask, image_patch_embeddings, patch_temperature, clip_temperature, p)
    loss.backward()
    image_top_grad = image_patch_embeddings.grad
    text_top_grad = text_embeddings.grad

    #propagate the grad down the backbones (accumulate_into_backbone)
    accumulate_into_image_backbones(image_backbone_lower, image_backbone_upper, image_embedder, image_batch, image_minibatch_size, image_top_grad, loss_weight)
    if text_backbone_upper is not None:
        accumulate_into_text_backbones(text_backbone_lower, text_backbone_upper, text_batch, text_minibatch_size, text_top_grad, loss_weight)

    #set training mode
    set_training_mode(image_backbone_upper,image_embedder,text_backbone_upper,is_training_dict)

    return loss.detach()


def add_to_backbone_gradients_smallbatch_multilabel_patch_alignment(image_backbone_lower, image_backbone_upper, image_embedder, text_backbone_lower, text_backbone_upper, patch_temperature, clip_temperature, image_batch, text_batch, text_mask, loss_weight, params):
    p = params

    #precompute the lower outputs
    if image_backbone_lower is not None:
        with torch.no_grad():
            image_batch = image_backbone_lower(image_batch)

    if text_backbone_lower is not None:
        with torch.no_grad():
            text_batch = text_backbone_lower(text_batch)

    image_patch_embeddings = special_forward(image_backbone_lower,[image_backbone_upper,image_embedder],image_batch)
    text_embeddings = special_forward(text_backbone_lower, [text_backbone_upper], text_batch)

    loss = compute_loss_from_embeddings(text_embeddings, text_mask, image_patch_embeddings, patch_temperature, clip_temperature, p)
    loss = loss_weight * loss
    loss.backward()

    return loss.detach()
