import os
import sys
import numpy as np
import pickle
import pprint
from tqdm import tqdm
from hand_eval import sanitize_labels, detect_adj_nouns

SUPPRESS_FN_BREAKDOWN = True

#strictness should be "correct", "mild", or "severe"
#return stats as dict with keys 'IOU', 'precision', 'recall', 'fn_adj_noun_prop', 'fn_noun_only_prop', 'fn_other_prop'
#other keys are 'recall_given_full_adj_noun', 'recall_given_noun_in_adj_noun', 'recall_given_other', which might map to None
#recall and the other 4 after it should add up to 1
#all these metrics are on 0-1 scale
def compute_extraction_accuracy_one_example(gt_labels, pred_labels, hand_eval_val, strictness):
    assert(strictness in ['correct', 'mild', 'severe'])
    gt_labels = sanitize_labels(gt_labels)
    pred_labels = sanitize_labels(pred_labels)
    adj_nouns = detect_adj_nouns(gt_labels)
    gt_labels = set(gt_labels)
    pred_labels = set(pred_labels)
    stats = {}
    stats['IOU'] = compute_IOU(gt_labels, pred_labels, hand_eval_val, strictness)
    stats['precision'] = compute_precision(gt_labels, pred_labels, hand_eval_val, strictness)
    for recall_metric_type in ['recall', 'fn_adj_noun_prop', 'fn_noun_only_prop', 'fn_other_prop', 'recall_given_full_adj_noun', 'recall_given_noun_in_adj_noun', 'recall_given_other']:
        stats[recall_metric_type] = compute_recall(gt_labels, pred_labels, adj_nouns, hand_eval_val, strictness, recall_metric_type)

    return stats

def compute_precision(gt_labels, pred_labels, hand_eval_val, strictness):
    assert(strictness in ['correct', 'mild', 'severe'])
    valids = gt_labels
    if strictness == 'correct':
        hand_types = ['correct']
    elif strictness == 'mild':
        hand_types = ['correct', 'mild']
    elif strictness == 'severe':
        hand_types = ['correct', 'mild', 'severe']
    else:
        assert(False)

    for hand_type in hand_types:
        valids = valids.union(set([p[1] for p in hand_eval_val[hand_type]]))

    return len(pred_labels.intersection(valids)) / len(pred_labels)

def compute_recall(gt_labels, pred_labels, adj_nouns, hand_eval_val, strictness, recall_metric_type):
    assert(strictness in ['correct', 'mild', 'severe'])
    valids = pred_labels
    if strictness == 'correct':
        hand_types = ['correct']
    elif strictness == 'mild':
        hand_types = ['correct', 'mild']
    elif strictness == 'severe':
        hand_types = ['correct', 'mild', 'severe']
    else:
        assert(False)

    for hand_type in hand_types:
        valids = valids.union(set([p[0] for p in hand_eval_val[hand_type]]))

    if recall_metric_type == 'recall':
        return len(gt_labels.intersection(valids)) / len(gt_labels)
    elif recall_metric_type == 'fn_adj_noun_prop':
        return len(set([p[0] for p in adj_nouns]) - valids) / len(gt_labels)
    elif recall_metric_type == 'fn_noun_only_prop':
        return len(set([p[1] for p in adj_nouns]) - valids) / len(gt_labels)
    elif recall_metric_type == 'fn_other_prop':
        subtracters = set([p[0] for p in adj_nouns]).union(set([p[1] for p in adj_nouns]))
        return len(gt_labels - subtracters - valids) / len(gt_labels)
    elif recall_metric_type == 'recall_given_full_adj_noun':
        full_adj_nouns = set([p[0] for p in adj_nouns])
        if len(full_adj_nouns) == 0:
            return None
        return len(full_adj_nouns.intersection(valids)) / len(full_adj_nouns)
    elif recall_metric_type == 'recall_given_noun_in_adj_noun':
        nouns_in_adj_nouns = set([p[1] for p in adj_nouns])
        if len(nouns_in_adj_nouns) == 0:
            return None
        return len(nouns_in_adj_nouns.intersection(valids)) / len(nouns_in_adj_nouns)
    elif recall_metric_type == 'recall_given_other':
        others = gt_labels - set([p[0] for p in adj_nouns]).union(set([p[1] for p in adj_nouns]))
        if len(others) == 0:
            return None
        return len(others.intersection(valids)) / len(others)
    else:
        assert(False)

#expect gt_labels, pred_labels to be a set and sanitized
#IOU is 0-1 scale
#assumes that no pred-gt links are overlapping with each other
#so we can just compute the intersection and union,
#and then each thingy from hand_eval_val will add one to the intersection and subtract two from the union
def compute_IOU(gt_labels, pred_labels, hand_eval_val, strictness):
    assert(strictness in ['correct', 'mild', 'severe'])
    intersection = len(gt_labels.intersection(pred_labels))
    union = len(gt_labels.union(pred_labels))
    if strictness == 'correct':
        hand_adjustment = len(hand_eval_val['correct'])
    elif strictness == 'mild':
        hand_adjustment = len(hand_eval_val['correct']) + len(hand_eval_val['mild'])
    elif strictness == 'severe':
        hand_adjustment = len(hand_eval_val['correct']) + len(hand_eval_val['mild']) + len(hand_eval_val['severe'])
    else:
        assert(False)

    return (intersection + hand_adjustment) / (union - 2 * hand_adjustment)

def get_disjoint_keys(gt_label_dict, metalabel_dict):
    filter_labels = []
    for k in sorted(metalabel_dict.keys()):
        filter_labels.extend(sanitize_labels(metalabel_dict[k]['labels']))

    filter_labels = set(filter_labels)
    disjoint_keys = []
    for k in sorted(gt_label_dict.keys()):
        labels = gt_label_dict[k]['labels']
        labels = set(sanitize_labels(labels))
        if len(labels.intersection(filter_labels)) == 0:
            disjoint_keys.append(k)

    return disjoint_keys

def compute_extraction_accuracy(gt_label_dict_filename, pred_label_dict_filename, hand_eval_dict_filename, metalabel_dict_filename, accuracy_filename):
    with open(gt_label_dict_filename, 'rb') as f:
        gt_label_dict = pickle.load(f)

    with open(pred_label_dict_filename, 'rb') as f:
        pred_label_dict = pickle.load(f)

    with open(hand_eval_dict_filename, 'rb') as f:
        hand_eval_dict = pickle.load(f)

    assert(sorted(pred_label_dict.keys()) == sorted(gt_label_dict.keys()))
    assert(sorted(hand_eval_dict.keys()) == sorted(gt_label_dict.keys()))

    with open(metalabel_dict_filename, 'rb') as f:
        metalabel_dict = pickle.load(f)

    accuracy = {'list' : {'correct' : {}, 'mild' : {}, 'severe' : {}}}
    for k in tqdm(sorted(gt_label_dict.keys())):
        for strictness in ['correct', 'mild', 'severe']:
            accuracy['list'][strictness][k] = compute_extraction_accuracy_one_example(gt_label_dict[k]['labels'], pred_label_dict[k]['labels'], hand_eval_dict[k], strictness)

    disjoint_keys = get_disjoint_keys(gt_label_dict, metalabel_dict)
    print('There are %d disjoint keys'%(len(disjoint_keys)))

    accuracy['avg'] = {'full' : {}, 'disjoint' : {}}
    for strictness in ['correct', 'mild', 'severe']:
        accuracy['avg']['full'][strictness] = {}
        accuracy['avg']['disjoint'][strictness] = {}

    for strictness in ['correct', 'mild', 'severe']:
        for metric_type in ['IOU', 'precision', 'recall', 'fn_adj_noun_prop', 'fn_noun_only_prop', 'fn_other_prop', 'recall_given_full_adj_noun', 'recall_given_noun_in_adj_noun', 'recall_given_other']:
            accuracy['avg']['full'][strictness][metric_type] = np.mean([accuracy['list'][strictness][k][metric_type] for k in sorted(gt_label_dict.keys()) if accuracy['list'][strictness][k][metric_type] is not None])
            accuracy['avg']['disjoint'][strictness][metric_type] = np.mean([accuracy['list'][strictness][k][metric_type] for k in disjoint_keys if accuracy['list'][strictness][k][metric_type] is not None])

    with open(accuracy_filename, 'wb') as f:
        pickle.dump(accuracy, f)

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(possibly_suppress_fn_breakdown(accuracy['avg']))

def possibly_suppress_fn_breakdown(acc_avg):
    if not SUPPRESS_FN_BREAKDOWN:
        return acc_avg

    acc_avg_out = {'full' : {'correct' : {}, 'mild' : {}, 'severe' : {}}, 'disjoint' : {'correct' : {}, 'mild' : {}, 'severe' : {}}}
    for disjointness in ['full', 'disjoint']:
        for strictness in ['correct', 'mild', 'severe']:
            for metric_type in ['IOU', 'precision', 'recall', 'fn_adj_noun_prop', 'fn_noun_only_prop', 'fn_other_prop', 'recall_given_full_adj_noun', 'recall_given_noun_in_adj_noun', 'recall_given_other']:
                if 'fn_' not in metric_type:
                    acc_avg_out[disjointness][strictness][metric_type] = acc_avg[disjointness][strictness][metric_type]

    return acc_avg_out


def usage():
    print('Usage: python compute_extraction_accuracy.py <gt_label_dict_filename> <pred_label_dict_filename> <hand_eval_dict_filename> <metalabel_dict_filename> <accuracy_filename>')

if __name__ == '__main__':
    compute_extraction_accuracy(*(sys.argv[1:]))
