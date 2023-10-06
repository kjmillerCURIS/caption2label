import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from PIL import Image
import torch
from torchvision.transforms import Resize, CenterCrop
from clip_training_utils import special_forward, compute_individual_label_similarities, aggregate_similarities
plt.rcParams.update({'font.size': 22})


IMSIZE = 224
PATCHSIZE = 16
PATCH_SIDELEN = IMSIZE // PATCHSIZE
NUM_VIT_TOKENS = PATCH_SIDELEN ** 2 + 1 #number of ViT tokens before the CLF token is possibly dropped
EMBEDDING_SIZE = 512
CMAP_PROB = 'viridis'
VMIN_PROB = 0.0
#VMAX_PROB = 1.0
VMAX_PROB = 0.75
CMAP_COSSIM = 'coolwarm'
#VMIN_COSSIM = -0.003
#VMAX_COSSIM = 0.003
VMIN_COSSIM = -0.1
VMAX_COSSIM = 0.1


#MEOW: for each training image, we want four plots (in UL-UR-LL-LR order):
#-1.) image (augmented, of course, and yes, put a buffer around it)
#-2.) patch_cossims heatmap
#-3.) patch_probs heatmap
#-4.) final cossim and caption (with "TRUE" or "FALSE") and label (with "SimpleAvg" or "KEPT" or "DROPPED")


#returns list of dicts, ranked in descending order by image-label similarity
#each dict contains the following keys:
#-"label_str"
#-"token_cossims"
#-"token_probs"
#-"image_cossim"
#-"inclusion_str" (can be "INCLUDED", "DROPPED", or "SimpleAvg")
#-"aggregate_cossim" (yes this will be same for all labels)
#-"caption_str" (yes this will be same for all labels)
def compute_things_of_interest(image_datum, text_datum, params, model):
    p = params
    with torch.no_grad():
        image_input = torch.unsqueeze(image_datum['image'].cuda(), 0)
        if model['image_backbone_lower'] is not None:
            image_input = model['image_backbone_lower'](image_input)

        image_patch_embeddings = special_forward(None, [model['image_backbone_upper'], model['image_embedder']], image_input)
        assert(image_patch_embeddings.shape == (1, NUM_VIT_TOKENS, EMBEDDING_SIZE)) #haven't dropped the CLF token yet
        text_input = torch.unsqueeze(text_datum['text'].cuda(), 0)
        if model['text_backbone_lower'] is not None:
            text_input = model['text_backbone_lower'](text_input)

        text_embeddings = special_forward(None, [model['text_backbone_upper']], text_input)
        assert(text_embeddings.shape == (1, p.max_num_labels, EMBEDDING_SIZE))
        image_cossims,pacl_extras = compute_individual_label_similarities(text_embeddings,image_patch_embeddings,model['patch_temperature'],p,return_extras=True)
        token_cossims_list = pacl_extras['patch_cossims']
        token_probs_list = pacl_extras['patch_probs']
        assert(token_cossims_list.shape == (1, NUM_VIT_TOKENS - int(p.drop_clf_token), 1, p.max_num_labels))
        assert(token_probs_list.shape == (1, NUM_VIT_TOKENS - int(p.drop_clf_token), 1, p.max_num_labels))
        aggregate_cossim, agg_extras = aggregate_similarities(torch.clone(image_cossims), torch.unsqueeze(text_datum['text_mask'].cuda(), 0), p, return_extras=True)
        assert(aggregate_cossim.shape == (1,1))
        aggregate_cossim = aggregate_cossim[0,0].item()
        is_included_list = agg_extras['is_included']
        assert(is_included_list.shape == (1, 1, p.max_num_labels))
        is_included_list = torch.squeeze(torch.squeeze(is_included_list, 1), 0)
        token_cossims_list = torch.squeeze(torch.squeeze(token_cossims_list, 2), 0)
        token_cossims_list = torch.permute(token_cossims_list, (1,0))
        token_probs_list = torch.squeeze(torch.squeeze(token_probs_list, 2), 0)
        token_probs_list = torch.permute(token_probs_list, (1,0))
        assert(image_cossims.shape == (1, 1, p.max_num_labels))
        image_cossims = torch.squeeze(torch.squeeze(image_cossims, 1), 0)
        things_of_interest = []
        assert(text_datum['text_mask'].shape == (p.max_num_labels,))
        for label_str, token_cossims, token_probs, image_cossim, is_included, text_mask_one in zip(text_datum['text_str_list'], token_cossims_list, token_probs_list, image_cossims, is_included_list, text_datum['text_mask']):
            if text_mask_one.item() == 0:
                continue

            thing_of_interest = {'caption_str' : text_datum['caption_str'], 'label_str' : label_str, 'token_cossims' : token_cossims.cpu().numpy(), 'token_probs' : token_probs.cpu().numpy(), 'image_cossim' : image_cossim.item(), 'aggregate_cossim' : aggregate_cossim}
            if p.aggregation_type == 'avg':
                thing_of_interest['inclusion_str'] = 'SimpleAvg'
            elif p.aggregation_type == 'avg_drop_percentile':
                thing_of_interest['inclusion_str'] = 'INCLUDED' if is_included.item() else 'DROPPED'
            else:
                assert(False)

            things_of_interest.append(thing_of_interest)

    return things_of_interest


def make_heatmap(arr, is_drop_clf):
    assert(arr.shape == (PATCH_SIDELEN ** 2 + 1 - int(is_drop_clf),))
    heatmap = np.zeros((PATCH_SIDELEN + 1 - int(is_drop_clf), PATCH_SIDELEN + 1 - int(is_drop_clf)))
    if is_drop_clf:
        heatmap[:,:] = np.reshape(arr, (PATCH_SIDELEN, PATCH_SIDELEN))
    else:
        heatmap[:-1,:-1] = np.reshape(arr[1:], (PATCH_SIDELEN, PATCH_SIDELEN))
        heatmap[-1,-1] = arr[0]

    heatmap = np.repeat(np.repeat(heatmap, PATCHSIZE, axis=0), PATCHSIZE, axis=1)
    return heatmap


def make_info_plot(ax_info, thing_of_interest, is_positive_pair, is_drop_clf):
    cmap = get_cmap(CMAP_COSSIM)
    norm = Normalize(vmin=VMIN_COSSIM, vmax=VMAX_COSSIM)
    image_cossim_color = cmap(norm(thing_of_interest['image_cossim']))[:3]
    aggregate_cossim_color = cmap(norm(thing_of_interest['aggregate_cossim']))[:3]
    heatmap_info = np.ones(((PATCH_SIDELEN + 1 - int(is_drop_clf)) * PATCHSIZE, (PATCH_SIDELEN + 1 - int(is_drop_clf)) * PATCHSIZE, 3), dtype=np.float32)
    for c in range(3):
        heatmap_info[100:120, 120:, c] = image_cossim_color[c]
        heatmap_info[140:160, 120:, c] = aggregate_cossim_color[c]
        ax_info.imshow(heatmap_info)
        ax_info.text(0, 20, 'caption (%s): "%s"'%({True : '+', False : '-'}[is_positive_pair], thing_of_interest['caption_str']))
        ax_info.text(0, 60, 'label: "%s"'%(thing_of_interest['label_str']))
        ax_info.text(0, 100, 'image_cossim (%s) = %.5f'%(thing_of_interest['inclusion_str'], thing_of_interest['image_cossim']))
        ax_info.text(0, 140, 'agg_cossim = %.5f'%(thing_of_interest['aggregate_cossim']))


#Note: This makes multiple plots
def plot_one_training_example(image_datum, text_datum, is_positive_pair, params, model, plot_prefix):
    p = params
    things_of_interest = compute_things_of_interest(image_datum, text_datum, p, model)
    things_of_interest = sorted(things_of_interest, key = lambda toi: toi['image_cossim'], reverse=True)
    with torch.no_grad():
        image_for_vis = torch.permute(image_datum['image_for_vis'], (1,2,0)).numpy()

    for i, thing_of_interest in enumerate(things_of_interest):
        plt.clf()
        heatmap_img = np.zeros(((PATCH_SIDELEN + 1 - int(p.drop_clf_token)) * PATCHSIZE, (PATCH_SIDELEN + 1 - int(p.drop_clf_token)) * PATCHSIZE, 3), dtype=image_for_vis.dtype)
        heatmap_img[:IMSIZE, :IMSIZE] = image_for_vis
        heatmap_cossims = make_heatmap(thing_of_interest['token_cossims'], p.drop_clf_token)
        heatmap_probs = make_heatmap(thing_of_interest['token_probs'], p.drop_clf_token)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[32, 24])
        axs = axs.ravel().tolist()
        axs[0].imshow(heatmap_img)
        axs[0].set_title('image')
        im_cossim = axs[1].imshow(heatmap_cossims, vmin=VMIN_COSSIM, vmax=VMAX_COSSIM, cmap=CMAP_COSSIM)
        axs[1].set_title('patch and clf token cossims')
        im_prob = axs[2].imshow(heatmap_probs, vmin=VMIN_PROB, vmax=VMAX_PROB, cmap=CMAP_PROB)
        axs[2].set_title('patch and clf token probs')
        cbar_cossim = plt.colorbar(im_cossim, ax=axs, location='right')
        cbar_cossim.set_label('cossim scale')
        cbar_prob = plt.colorbar(im_prob, ax=axs, location='left')
        cbar_prob.set_label('prob scale')
        make_info_plot(axs[3], thing_of_interest, is_positive_pair, p.drop_clf_token)
        plt.savefig(plot_prefix + '-' + {True : 'pos_pair', False : 'neg_pair'}[is_positive_pair] + '-label%02d'%(i) + '.png')
        plt.clf()
        plt.close(fig)


#def toy_visualize_heatmap():
#    plt.rcParams.update({'font.size': 22})
#    kitty = np.array(CenterCrop(IMSIZE)(Resize(IMSIZE)(Image.open('cat.jpeg'))))
#    heatmap_kitty = np.zeros((kitty.shape[0] + PATCHSIZE, kitty.shape[1] + PATCHSIZE, 3), dtype=kitty.dtype)
#    heatmap_kitty[:IMSIZE, :IMSIZE] = kitty
#    meows = np.random.randn(PATCH_SIDELEN**2+1, 2)
#    cossims = meows[:,0] / np.sqrt(np.sum(np.square(meows), axis=1))
#    temp = 100
#    adj_cossims = cossims - np.amax(cossims) #this is equivalent to multiplying by temp first, so it's valid
#    probs = np.exp(temp*adj_cossims) / np.sum(np.exp(temp*adj_cossims))
#    heatmap_cossims = np.zeros((PATCH_SIDELEN+1, PATCH_SIDELEN+1))
#    patch_cossims = np.reshape(cossims[1:], (PATCH_SIDELEN, PATCH_SIDELEN))
#    clf_cossim = cossims[0]
#    heatmap_cossims[:-1,:-1] = patch_cossims
#    heatmap_cossims[-1,-1] = clf_cossim
#    heatmap_cossims = np.repeat(np.repeat(heatmap_cossims, PATCHSIZE, axis=0), PATCHSIZE, axis=1)
#    patch_probs = np.reshape(probs[1:], (PATCH_SIDELEN, PATCH_SIDELEN))
#    clf_prob = probs[0]
#    heatmap_probs = np.zeros((PATCH_SIDELEN+1, PATCH_SIDELEN+1))
#    heatmap_probs[:-1,:-1] = patch_probs
#    heatmap_probs[-1,-1] = clf_prob
#    heatmap_probs = np.repeat(np.repeat(heatmap_probs, PATCHSIZE, axis=0), PATCHSIZE, axis=1)
#    _, axs = plt.subplots(nrows=2, ncols=2, figsize=[32, 24])
#    axs = axs.ravel().tolist()
#    axs[0].imshow(heatmap_kitty)
#    axs[0].set_title('image')
#    axs[3].imshow(np.ones_like(heatmap_kitty) * (255 if heatmap_kitty.dtype == 'uint8' else 1))
#    im_cossim = axs[1].imshow(heatmap_cossims, vmin=-1, vmax=1, cmap='coolwarm')
#    axs[1].set_title('patch and clf token cossims')
#    im_prob = axs[2].imshow(heatmap_probs, vmin=0, vmax=0.1, cmap='viridis')
#    axs[2].set_title('patch and clf token probs')
#    cbar_cossim = plt.colorbar(im_cossim, ax=axs, location='right')
#    cbar_cossim.set_label('cossim scale')
#    cbar_prob = plt.colorbar(im_prob, ax=axs, location='left')
#    cbar_prob.set_label('prob scale')
#    plt.savefig('toy_heatmap.png')
#    plt.clf()
#
#
#if __name__ == '__main__':
#    toy_visualize_heatmap()
