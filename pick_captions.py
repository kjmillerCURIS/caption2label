import os
import sys
import pickle
import random
from hand_label_captions import load_captions

NUM_SUPPORT = 32
NUM_EVAL = 300
RANDOM_SEED = 0

def pick_captions(caption_csv_filename, support_filename, eval_filename, random_seed):
    random_seed = int(random_seed)

    caption_dict = load_captions(caption_csv_filename)
    all_key_list = sorted(caption_dict.keys())
    support_key_list = random.sample(all_key_list, NUM_SUPPORT)
    eval_key_list = random.sample([k for k in all_key_list if k not in support_key_list], NUM_EVAL)

    with open(support_filename, 'wb') as f:
        pickle.dump(support_key_list, f)

    with open(eval_filename, 'wb') as f:
        pickle.dump(eval_key_list, f)

def usage():
    print('Usage: python pick_captions.py <caption_csv_filename> <support_filename> <eval_filename> <random_seed>')

if __name__ == '__main__':
    pick_captions(*(sys.argv[1:]))
