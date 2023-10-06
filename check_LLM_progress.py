import os
import sys
import pickle
from hand_label_captions import load_captions

def check_LLM_progress(caption_csv_filename, output_prefix, stride):
    stride = int(stride)

    caption_dict = load_captions(caption_csv_filename)
    N = len(caption_dict)

    for i in range(stride):
        N_i = len([j for j in range(N) if j % stride == i % stride])
        with open(output_prefix + '_part%d.pkl'%(i), 'rb') as f:
            d = pickle.load(f)

        n_i = len(d)
        print('%d: %d / %d'%(i, n_i, N_i))

def usage():
    print('Usage: python check_LLM_progress.py <caption_csv_filename> <output_prefix> <stride>')

if __name__ == '__main__':
    check_LLM_progress(*(sys.argv[1:]))
