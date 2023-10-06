import os
import sys
import numpy as np
import pickle


PREVALENCE_THRESHOLD = 100


#np array of bools, same order as classnames
def get_class_mask(class_prevalence_dict_filename, dataset_name):
    with open(class_prevalence_dict_filename, 'rb') as f:
        d = pickle.load(f)

    assert(len(d[dataset_name]['classnames']) == len(d[dataset_name]['prevalences']))
    return np.array([v >= PREVALENCE_THRESHOLD for v in d[dataset_name]['prevalences']])
