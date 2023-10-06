import os
import sys

SINGLE_TEMPLATE = {}
SINGLE_TEMPLATE['Flowers102'] = 'a photo of a {}, a type of flower.'
SINGLE_TEMPLATE['Food101'] = 'a photo of {}, a type of food.'
SINGLE_TEMPLATE['UCF101'] = 'a photo of a person doing {}.'
SINGLE_TEMPLATE['ImageNet1K'] = 'a photo of a {}.'
SINGLE_TEMPLATE['DTD'] = 'a photo of a {} texture.' #CoOp just had '{} texture', but I'd rather use something that CLIP actually used in their ensemble
