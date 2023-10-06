import os
import sys
import numpy as np
import torch
from torch import nn


class CustomVisionTransformerLaCLIP(nn.Module):

    def __init__(self, laclip_model, num_learnable_layers, lower_or_upper):
        super().__init__()
        assert(lower_or_upper in ['lower', 'upper'])
        assert(num_learnable_layers >= 0)
        their_num_layers = len(list(laclip_model.visual.blocks.children()))
        assert(num_learnable_layers <= their_num_layers)
        their_dtype = laclip_model.visual.patch_embed.proj.weight.dtype
        self.dtype = their_dtype
        self.lower_or_upper = lower_or_upper
        if self.lower_or_upper == 'lower':
            assert(num_learnable_layers < their_num_layers)
            self.is_whole = (num_learnable_layers <= 0)
            self._init_before_transformer(laclip_model)
            all_blocks = list(laclip_model.visual.blocks.children())
            self.blocks = nn.Sequential(*all_blocks[:their_num_layers - num_learnable_layers])
            if self.is_whole:
                self._init_after_transformer(laclip_model)

        elif self.lower_or_upper == 'upper':
            assert(num_learnable_layers > 0)
            self.is_whole = (num_learnable_layers >= their_num_layers)
            if self.is_whole:
                self._init_before_transformer(laclip_model)

            all_blocks = list(laclip_model.visual.blocks.children())
            self.blocks = nn.Sequential(*all_blocks[their_num_layers - num_learnable_layers:])
            self._init_after_transformer(laclip_model)
        else:
            assert(False)

    def _init_before_transformer(self, laclip_model):
        self.patch_embed = laclip_model.visual.patch_embed
        self.cls_token = laclip_model.visual.cls_token
        self.pos_embed = laclip_model.visual.pos_embed
        self.pos_drop = laclip_model.visual.pos_drop
        self.patch_drop = laclip_model.visual.patch_drop
        self.norm_pre = laclip_model.visual.norm_pre

    def _init_after_transformer(self, laclip_model):
        self.norm = laclip_model.visual.norm
        self.fc_norm = laclip_model.visual.fc_norm
        self.head_drop = laclip_model.visual.head_drop

    def _pos_embed(self, x):
        assert(self.cls_token is not None)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def before_transformer(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        return x

    def after_transformer(self, x: torch.Tensor):
        #in practice, one of norm and fc_norm will actually normalize, and the other will be nn.Identity()
        x = self.norm(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        #no need for self.head(x), because we know that it will be nn.Identity()
        return x

    def run_transformer(self, x: torch.Tensor):
        return self.blocks(x) #shouldn't need to switch around the dimensions this time...

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
