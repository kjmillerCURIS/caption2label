import os
import sys
import glob
#from Levenshtein import distance
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from eval_utils import DATASET_DICT, INFERENCE_CONTEXT_LENGTH
from hand_eval import sanitize_labels, sanitize_label
from write_to_log_file import write_to_log_file


CLIP_MODEL_TYPE = 'ViT-B/16'
#LEVENSHTEIN_THRESHOLD = 3
#LEVENSHTEIN_THRESHOLD_NORMALIZED = 0.333 #distance can be at most this much times the length of the longer word (so "cat" can't become "dog")
#VISUALIZED_PREVALENCE_LINES = [50, 100, 200]
VISUALIZED_PREVALENCE_LINES = [100]


#def is_close(classname, my_label):
#    my_dist = distance(classname, my_label, score_cutoff=LEVENSHTEIN_THRESHOLD+1)
#    return my_dist <= min(LEVENSHTEIN_THRESHOLD, LEVENSHTEIN_THRESHOLD_NORMALIZED * max(len(classname), len(my_label)))


#we'll just do exact matches for now
#but if you wanted inexact matches, you could 
def compute_class_prevalence_one_dataset(dataset_parent_dir, dataset_name, label_counter, class_prevalence_plot_prefix):
    dataset = DATASET_DICT[dataset_name](dataset_parent_dir, INFERENCE_CONTEXT_LENGTH, CLIP_MODEL_TYPE)
    out = {'classnames' : dataset.classnames}
    out['prevalences'] = []
    for raw_classname in tqdm(dataset.classnames):
        classname = sanitize_label(raw_classname)
        total_count = 0
        for my_label in sorted(label_counter.keys()):
#            if my_label == classname or is_close(classname, my_label):
            if my_label == classname:
                total_count += label_counter[my_label]
#                if my_label != classname:
#                    write_to_log_file('%s (%s) >==< %s'%(classname, raw_classname, my_label))

        out['prevalences'].append(total_count)

    sorted_prevalences = sorted(out['prevalences'], reverse=True)
#    sorted_prevalences = [v for v in sorted_prevalences if v > 0]
    plt.clf()
    plt.scatter(np.arange(len(sorted_prevalences)), sorted_prevalences)
    plt.vlines(np.arange(len(sorted_prevalences)), 0, sorted_prevalences, linestyles='dashed')
    my_xlim = plt.xlim()
    plt.hlines(VISUALIZED_PREVALENCE_LINES, my_xlim[0], my_xlim[1], linestyles='dashed')
    plt.xlim(my_xlim)
    plt.xlabel('rank')
    plt.ylabel('prevalence')
    plt.title('%s classname prevalence in training set'%(dataset_name))
    plt.savefig(class_prevalence_plot_prefix + '_' + dataset_name)
    plt.clf()
    return out


def build_label_counter(text_filename_prefix):
    text_filenames = sorted(glob.glob(os.path.join(text_filename_prefix + '_part*.pkl')))
    assert(len(text_filenames) == 16)
    label_counter = {}
    for text_filename in text_filenames:
        with open(text_filename, 'rb') as f:
            d = pickle.load(f)

        for k in sorted(d.keys()):
            my_labels = sanitize_labels(d[k]['labels'])
            for my_label in my_labels:
                if my_label not in label_counter:
                    label_counter[my_label] = 0

                label_counter[my_label] = label_counter[my_label] + 1

    return label_counter


def compute_class_prevalence(dataset_parent_dir, text_filename_prefix, class_prevalence_dict_filename, class_prevalence_plot_prefix):
    os.makedirs(os.path.dirname(class_prevalence_plot_prefix), exist_ok=True)
    write_to_log_file('building label_counter...')
    label_counter = build_label_counter(text_filename_prefix)
    write_to_log_file('done building label_counter')
    class_prevalence_dict = {}
    for dataset_name in sorted(DATASET_DICT.keys()):
        class_prevalence_dict[dataset_name] = compute_class_prevalence_one_dataset(dataset_parent_dir, dataset_name, label_counter, class_prevalence_plot_prefix)

    with open(class_prevalence_dict_filename, 'wb') as f:
        pickle.dump(class_prevalence_dict, f)


def usage():
    print('Usage: python compute_class_prevalence.py <dataset_parent_dir> <text_filename_prefix> <class_prevalence_dict_filename> <class_prevalence_plot_prefix>')


if __name__ == '__main__':
    compute_class_prevalence(*(sys.argv[1:]))
