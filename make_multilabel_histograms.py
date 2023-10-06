import os
import sys
import clip
import glob
import numpy as np
import pickle
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from hand_eval import sanitize_labels


def process_one(labels):
    labels = sorted(set(sanitize_labels(labels)))
    text = clip.tokenize(labels)
    num_tokens_mult = torch.argmax(text, dim=1) + 1
    num_tokens_mult = num_tokens_mult.numpy().tolist()
    return len(labels), num_tokens_mult


def make_multilabel_histograms(text_filename_prefix, num_labels_histogram_filename, num_tokens_histogram_filename):
    num_labels_list = []
    num_tokens_list = []
    text_filenames = sorted(glob.glob(text_filename_prefix + '_part*.pkl'))
    for text_filename in tqdm(text_filenames):
        with open(text_filename, 'rb') as f:
            d = pickle.load(f)

        for k in tqdm(sorted(d.keys())):
            num_labels, num_tokens_mult = process_one(d[k]['labels'])
            num_labels_list.append(num_labels)
            num_tokens_list.extend(num_tokens_mult)
            print('99th percentile num_tokens = %s'%(str(np.percentile(num_tokens_list, 99))))
            print('max num_tokens = %s'%(str(np.amax(num_tokens_list))))
            print('95th percentile num_labels = %s'%(str(np.percentile(num_labels_list, 95))))
            print('max num_labels = %s'%(str(np.amax(num_labels_list))))

    plt.clf()
    plt.xlabel('num_labels')
    plt.ylabel('freq')
    plt.hist(num_labels_list, bins=16)
    plt.savefig(num_labels_histogram_filename)
    plt.clf()
    plt.xlabel('num_tokens')
    plt.ylabel('freq')
    plt.hist(num_tokens_list, bins=8)
    plt.savefig(num_tokens_histogram_filename)
    plt.clf()


def usage():
    print('Usage: python make_multilabel_histograms.py <text_filename_prefix> <num_labels_histogram_filename> <num_tokens_histogram_filename>')


if __name__ == '__main__':
    make_multilabel_histograms(*(sys.argv[1:]))
