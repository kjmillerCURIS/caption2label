import os
import sys
import torch
from collections import OrderedDict
from laclip_models import CLIP_VITB16


#yes, this is harcoded (for now)
STARTPOINT_FILENAME = '/usr3/graduate/nivek/data/vislang-domain-exploration-data/caption2label-data/pretrained_checkpoints/cc3m_clip.pt'


def load_laclip_startpoint():
    ckpt = torch.load(STARTPOINT_FILENAME, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    laclip_model = CLIP_VITB16(rand_embed=False)
    laclip_model.cuda()
    laclip_model.load_state_dict(state_dict, strict=True)
    return laclip_model
