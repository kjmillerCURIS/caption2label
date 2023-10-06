import os
import sys
from experiment_params.params import grab_params
from clip_backbone_utils import get_backbones_with_patch_alignment


if __name__ == '__main__':
    p = grab_params('PatchAlignmentMultilabelParams')
    image_backbone_lower, image_backbone_upper, image_embedder, text_backbone_lower, text_backbone_upper, patch_temperature, clip_temperature = get_backbones_with_patch_alignment(p, is_laclip=True)
    import pdb
    pdb.set_trace()
