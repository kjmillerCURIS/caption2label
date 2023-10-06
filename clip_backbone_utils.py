import os
import sys
import clip
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from custom_vision_transformer_laclip import CustomVisionTransformerLaCLIP
from load_laclip_startpoint import load_laclip_startpoint


#get it, cuz it applies the temperature?
class Incubator(nn.Module):
    def __init__(self, logit_scale):
        super().__init__()
        self.logit_scale = logit_scale #specifically have to have "logit_scale" in the name

    def forward(self, x):
        return self.logit_scale.to(torch.float16).exp() * x


def build_efficient_attn_mask(max_num_tokens):
    mask = torch.empty(max_num_tokens, max_num_tokens)
    mask.fill_(float('-inf'))
    mask.triu_(1)  # zero out the lower diagonal
    return mask


class EfficientResidualAttentionBlock(nn.Module):
    def __init__(self, clip_resblock, attn_mask):
        super().__init__()
        self.attn = clip_resblock.attn
        self.ln_1 = clip_resblock.ln_1
        self.mlp = clip_resblock.mlp
        self.ln_2 = clip_resblock.ln_2
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        meow = self.attention(self.ln_1(x))
        x = x + meow
        x = x + self.mlp(self.ln_2(x))
        return x


#will be able to handle 2D and 3D texts (as well as 3D and 4D intermediates)
class CustomTextTransformer(nn.Module):

    #clip_model can be from CLIP or LaCLIP, but please use is_laclip flag to specify which one it is
    def __init__(self, clip_model, num_learnable_layers, lower_or_upper, max_num_tokens, is_laclip=False):
        super().__init__()
        assert(lower_or_upper in ['lower', 'upper'])
        assert(num_learnable_layers >= 0)
        assert(num_learnable_layers <= clip_model.transformer.layers)
        their_dtype = clip_model.visual.patch_embed.proj.weight.dtype if is_laclip else clip_model.dtype
        self.dtype = their_dtype
        self.lower_or_upper = lower_or_upper
        self.context_length = max_num_tokens
        self.vocab_size = clip_model.vocab_size
        attn_mask = build_efficient_attn_mask(max_num_tokens)
        if self.lower_or_upper == 'lower':
            assert(num_learnable_layers < clip_model.transformer.layers)
            self.is_whole = (num_learnable_layers <= 0)
            self.token_embedding = clip_model.token_embedding
            self.positional_embedding = clip_model.positional_embedding
            all_resblocks = list(clip_model.transformer.resblocks.children())
            my_resblocks = []
            for clip_resblock in all_resblocks[:clip_model.transformer.layers - num_learnable_layers]:
                my_resblocks.append(EfficientResidualAttentionBlock(clip_resblock, attn_mask))

            self.transformer = nn.Sequential(*my_resblocks)
            if self.is_whole:
                self.ln_final = clip_model.ln_final
                self.text_projection = clip_model.text_projection

        elif self.lower_or_upper == 'upper':
            assert(num_learnable_layers > 0)
            self.is_whole = (num_learnable_layers >= clip_model.transformer.layers)
            if self.is_whole:
                self.token_embedding = clip_model.token_embedding
                self.positional_embedding = clip_model.positional_embedding

            all_resblocks = list(clip_model.transformer.resblocks.children())
            my_resblocks = []
            for clip_resblock in all_resblocks[-num_learnable_layers:]:
                my_resblocks.append(EfficientResidualAttentionBlock(clip_resblock, attn_mask))

            self.transformer = nn.Sequential(*my_resblocks)
            self.ln_final = clip_model.ln_final
            self.text_projection = clip_model.text_projection
        else:
            assert(False)

    def change_context_length(self, new_context_length):
        self.context_length = new_context_length
        attn_mask = build_efficient_attn_mask(new_context_length)
        my_resblocks = list(self.transformer.children())
        my_resblocks = [EfficientResidualAttentionBlock(my_resblock, attn_mask) for my_resblock in my_resblocks]
        self.transformer = nn.Sequential(*my_resblocks)

    def before_transformer(self, text: torch.Tensor):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)[:self.context_length,:]
        return x

    def after_transformer(self, x: torch.Tensor, text: torch.Tensor):
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def run_transformer(self, x: torch.Tensor):
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x

    def _flatten_stuff_if_needed(self, stuff, flat_ndims):
        if len(stuff.shape) == flat_ndims+1:
            return torch.flatten(stuff, start_dim=0, end_dim=1), stuff.shape[:2]
        else:
            assert(len(stuff.shape) == flat_ndims)
            return stuff, None
    
    def _unflatten_stuff_if_needed(self, stuff, unflat_shape, flat_ndims):
        assert(len(stuff.shape) == flat_ndims) #either it already came in flat (and should come out that way), or it's flat because we flattened it
        if unflat_shape is None:
            return stuff
        else:
            return torch.unflatten(stuff, dim=0, sizes=unflat_shape)

    def _flatten_text_if_needed(self, text):
        return self._flatten_stuff_if_needed(text, 2)
    
    def _unflatten_text_if_needed(self, text, unflat_shape):
        return self._unflatten_stuff_if_needed(text, unflat_shape, 2)
    
    def _flatten_X_if_needed(self, X):
        return self._flatten_stuff_if_needed(X, 3)
    
    def _unflatten_X_if_needed(self, X, unflat_shape):
        return self._unflatten_stuff_if_needed(X, unflat_shape, 3)
    
    def _unflatten_outX_if_needed(self, outX, unflat_shape):
        return self._unflatten_stuff_if_needed(outX, unflat_shape, 2)

    #hello, I am the CustomTextTransformer!
    #if I have the bottom, then text_and_maybe_x is just text!
    #if I don't have the bottom, then text_and_maybe_x is (text, x)!
    #if I have the top, then I return x!
    #if I don't have the top, then I return (text, x)!
    def forward(self, text_and_maybe_x):
        if self.lower_or_upper == 'lower':
            text = text_and_maybe_x
            text, unflat_shape = self._flatten_text_if_needed(text)
            x = self.before_transformer(text)
            x = self.run_transformer(x)
            if self.is_whole:
                x = self.after_transformer(x, text)
                x = self._unflatten_outX_if_needed(x, unflat_shape)
                return x
            else:
                text = self._unflatten_text_if_needed(text, unflat_shape)
                x = self._unflatten_X_if_needed(x, unflat_shape)
                return (text, x)

        elif self.lower_or_upper == 'upper':
            if self.is_whole:
                text = text_and_maybe_x
                text, unflat_shape = self._flatten_text_if_needed(text)
                x = self.before_transformer(text)
            else:
                text, x = text_and_maybe_x
                text, unflat_shape = self._flatten_text_if_needed(text)
                x, unflat_shape_x = self._flatten_X_if_needed(x)
                assert(unflat_shape_x == unflat_shape)

            x = self.run_transformer(x)
            x = self.after_transformer(x, text)
            x = self._unflatten_outX_if_needed(x, unflat_shape)
            return x
        else:
            assert(False)


#use this class to get both the lower and the upper part
#if a part is gonna be None, don't even call this class, just make it None
#we treat the first resblock as being fused to everything below it, and the last resblock as being fused to everything above it
class CustomVisionTransformer(nn.Module):
    def __init__(self, clip_model, num_learnable_layers, lower_or_upper):
        super().__init__()
        assert(lower_or_upper in ['lower', 'upper'])
        assert(num_learnable_layers >= 0)
        assert(num_learnable_layers <= clip_model.visual.transformer.layers)
        self.dtype = clip_model.dtype
        self.lower_or_upper = lower_or_upper
        if self.lower_or_upper == 'lower':
            assert(num_learnable_layers < clip_model.visual.transformer.layers)
            self.is_whole = (num_learnable_layers <= 0)
            self.conv1 = clip_model.visual.conv1
            self.class_embedding = clip_model.visual.class_embedding
            self.positional_embedding = clip_model.visual.positional_embedding
            self.ln_pre = clip_model.visual.ln_pre
            all_resblocks = list(clip_model.visual.transformer.resblocks.children())
            self.transformer = nn.Sequential(*all_resblocks[:clip_model.visual.transformer.layers - num_learnable_layers])
            if self.is_whole:
                self.ln_post = clip_model.visual.ln_post

        elif self.lower_or_upper == 'upper':
            assert(num_learnable_layers > 0)
            self.is_whole = (num_learnable_layers >= clip_model.visual.transformer.layers)
            if self.is_whole:
                self.conv1 = clip_model.visual.conv1
                self.class_embedding = clip_model.visual.class_embedding
                self.positional_embedding = clip_model.visual.positional_embedding
                self.ln_pre = clip_model.visual.ln_pre

            all_resblocks = list(clip_model.visual.transformer.resblocks.children())
            self.transformer = nn.Sequential(*all_resblocks[-num_learnable_layers:])
            self.ln_post = clip_model.visual.ln_post
        else:
            assert(False)

    def before_transformer(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        return x

    def after_transformer(self, x: torch.Tensor):
        #yes, this means that patch embeddings and CLS embedding are all there, as a sequence of tokens
        #LayerNorm should have no problem taking a 3D instead of a 2D tensor
        x = self.ln_post(x)
        return x

    def run_transformer(self, x: torch.Tensor):
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x

    def forward(self, x: torch.Tensor):
        if self.lower_or_upper == 'lower':
            x = self.before_transformer(x.type(self.dtype))
            x = self.run_transformer(x)
            if self.is_whole:
                x = self.after_transformer(x)

        elif self.lower_or_upper == 'upper':
            if self.is_whole:
                x = self.before_transformer(x.type(self.dtype))

            x = self.run_transformer(x)
            x = self.after_transformer(x)
        else:
            assert(False)

        return x


class ImageEmbedder(nn.Module):

    #clip_model can be from CLIP or LaCLIP, but you should use is_laclip flag to indicate which one it is
    def __init__(self, clip_model, params, is_laclip=False):
        super().__init__()
        p = params

        mid_embedding_size = p.image_mid_embedding_size
        if is_laclip:
            their_proj = clip_model.image_projection
        else:
            their_proj = clip_model.visual.proj

        pre_embedding_size = their_proj.shape[0]
        embedding_size = their_proj.shape[1]

        main_fc1 = nn.Linear(pre_embedding_size, mid_embedding_size)
        main_fc2 = nn.Linear(mid_embedding_size, embedding_size)
        resid_fc = nn.Linear(pre_embedding_size, embedding_size)

        #idea is to make this initially behave like the CLIP image backbone's projection, in hopes of good initial performance
        if p.image_embedder_smart_start:
            alpha = p.image_embedder_smart_start_alpha
            with torch.no_grad():
                resid_fc.weight.data = alpha * their_proj.data.t().detach().cpu() + (1-alpha) * resid_fc.weight.data.detach()
                resid_fc.bias.data = (1-alpha) * resid_fc.bias.data.detach()
                main_fc1.weight.data = np.sqrt(1-alpha) * main_fc1.weight.data.detach()
                main_fc1.bias.data = np.sqrt(1-alpha) * main_fc1.bias.data.detach()
                main_fc2.weight.data = np.sqrt(1-alpha) * main_fc2.weight.data.detach()
                main_fc2.bias.data = (1-alpha) * main_fc2.bias.data.detach()

        self.main_branch = nn.Sequential(*[main_fc1, nn.ReLU(), main_fc2])
        self.resid_branch = resid_fc

    def forward(self, image_patch_pre_embeddings: torch.Tensor):
        assert(len(image_patch_pre_embeddings.shape) == 3)
        image_patch_pre_embeddings_flat = torch.flatten(image_patch_pre_embeddings, start_dim=0, end_dim=1)
        image_patch_embeddings_flat = self.main_branch(image_patch_pre_embeddings_flat) + self.resid_branch(image_patch_pre_embeddings_flat)
        image_patch_embeddings = torch.unflatten(image_patch_embeddings_flat, dim=0, sizes=image_patch_pre_embeddings.shape[:2])
        return image_patch_embeddings


def load_either_clip_startpoint(clip_model_type, is_laclip=False):
    if is_laclip:
        clip_model = load_laclip_startpoint()
    else:
        clip_model, _ = clip.load(clip_model_type, device='cuda')

    return clip_model


#will return backbones and image-embedder, former of which will be from OpenAI checkpoint, latter will be our own creation with randomly initialized weights
def get_backbones_with_patch_alignment(params, is_laclip=False):
    p = params

    custom_vision_transformer_class = CustomVisionTransformerLaCLIP if is_laclip else CustomVisionTransformer
    clip_model = load_either_clip_startpoint(p.clip_model_type, is_laclip=is_laclip)
    clip_model.float()
    clip_temperature = Incubator(clip_model.logit_scale)
    patch_temperature = Incubator(nn.Parameter(torch.ones([]) * p.init_patch_temperature))
    clip_temperature, patch_temperature = clip_temperature.cuda(), patch_temperature.cuda()
    their_num_layers = len(list(clip_model.visual.blocks.children())) if is_laclip else clip_model.visual.transformer.layers
    if p.image_num_learnable_layers < their_num_layers:
        image_backbone_lower = custom_vision_transformer_class(clip_model, p.image_num_learnable_layers, 'lower')
    else:
        image_backbone_lower = None

    if p.image_num_learnable_layers > 0:
        image_backbone_upper = custom_vision_transformer_class(clip_model, p.image_num_learnable_layers, 'upper')
    else:
        image_backbone_upper = None

    image_embedder = ImageEmbedder(clip_model, p, is_laclip=is_laclip)
    image_embedder = image_embedder.cuda()
    if p.text_num_learnable_layers < clip_model.transformer.layers:
        text_backbone_lower = CustomTextTransformer(clip_model,p.text_num_learnable_layers,'lower',p.max_num_tokens,is_laclip=is_laclip)
    else:
        text_backbone_lower = None

    if p.text_num_learnable_layers > 0:
        text_backbone_upper = CustomTextTransformer(clip_model,p.text_num_learnable_layers,'upper',p.max_num_tokens,is_laclip=is_laclip)
    else:
        text_backbone_upper = None

    return image_backbone_lower, image_backbone_upper, image_embedder, text_backbone_lower, text_backbone_upper, patch_temperature, clip_temperature
