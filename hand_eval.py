import os
import sys
import pandas as pd
import pickle
import random

#make lowercase, remove punctuation, remove extraneous spaces
def sanitize_label(label):
    label = label.lower()
    label = label.translate({ord(c) : None for c in '.,"\'?!*&^%$#@)(}{][/\\:;<>~-_=+'})
    label = ' '.join(label.split())
    return label

#labels should be a list, and we'll return a list (possibly shorter if some labels reduce to empty strings)
def sanitize_labels(labels):
    labels = [sanitize_label(label) for label in labels]
    labels = [label for label in labels if label != '']
    return labels

#see if (labelA, labelB) is an (adj_noun, noun_only) pair
def is_adj_noun(labelA, labelB):
    wordsA = labelA.split(' ')
    wordsB = labelB.split(' ')
    if len(wordsA) < 2 or len(wordsA) <= len(wordsB):
        return False

    offsetA = len(wordsA) - len(wordsB)
    return all([wA == wB for wA, wB in zip(wordsA[offsetA:], wordsB)])

#labels should be output of sanitize_labels()
#we'll return a list of pairs, in (adj_noun, noun_only) order
def detect_adj_nouns(labels):
    if len(labels) < 2:
        return []

    ret = []
    for i in range(len(labels) - 1):
        labelA = labels[i]
        labelB = labels[i+1]
        if is_adj_noun(labelA, labelB):
            ret.append((labelA, labelB))

    return ret

def print_status(cur_t, total_t, caption, gt_labels, pred_labels, hand_eval_val, other_hand_eval_val=None):
    gt_labels = sanitize_labels(gt_labels)
    pred_labels = sanitize_labels(pred_labels)
    adj_nouns = detect_adj_nouns(gt_labels)
    print('current position: %d/%d'%(cur_t, total_t))
    print('caption: "%s"'%(caption))
    print('gt: %s'%(str(gt_labels)))
    print('pred: %s'%(str(pred_labels)))
    gt_labels = set(gt_labels)
    pred_labels = set(pred_labels)
    print('intersection: %s'%(str(sorted(gt_labels.intersection(pred_labels)))))
    print('false negatives: %s'%(str(sorted(gt_labels - pred_labels))))
    print('false positives: %s'%(str(sorted(pred_labels - gt_labels))))
    print('correct: %s'%(str(sorted(hand_eval_val['correct']))))
    print('mild: %s'%(str(sorted(hand_eval_val['mild']))))
    print('severe: %s'%(str(sorted(hand_eval_val['severe']))))
    print('correct/mild/severe should be pairs in (gt, pred) order')
    print('adj_nouns: %s'%(str(adj_nouns)))
    if other_hand_eval_val is not None:
        print('SUGGESTED: %s'%(str(other_hand_eval_val)))

#will modify hand_eval_dict, which will map to dict with keys 'correct', 'mild', 'severe'
#another script will compute stats based on gt_label-dict, pred_label_dict, and hand_eval_dict
def hand_eval(gt_label_dict_filename, pred_label_dict_filename, hand_eval_dict_filename, random_seed, other_hand_eval_dict_filename=None):
    if random_seed == 'None':
        random_seed = None
    else:
        random_seed = int(random_seed)

    with open(gt_label_dict_filename, 'rb') as f:
        gt_label_dict = pickle.load(f)

    with open(pred_label_dict_filename, 'rb') as f:
        pred_label_dict = pickle.load(f)

    assert(sorted(pred_label_dict.keys()) == sorted(gt_label_dict.keys()))

    if not os.path.exists(hand_eval_dict_filename):
        hand_eval_dict = {k : {'correct' : set([]), 'mild' : set([]), 'severe' : set([])} for k in sorted(gt_label_dict.keys())}
    else:
        with open(hand_eval_dict_filename, 'rb') as f:
            hand_eval_dict = pickle.load(f)

    other_hand_eval_dict = None
    if other_hand_eval_dict_filename is not None:
        with open(other_hand_eval_dict_filename, 'rb') as f:
            other_hand_eval_dict = pickle.load(f)

    key_list = sorted(gt_label_dict.keys())
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(key_list)

    cur_t = 0
    while True:
        my_key = key_list[cur_t]
        caption = gt_label_dict[my_key]['caption']
        gt_labels = gt_label_dict[my_key]['labels']
        pred_labels = pred_label_dict[my_key]['labels']
        other_hand_eval_val = None
        if other_hand_eval_dict is not None:
            other_hand_eval_val = other_hand_eval_dict[my_key]

        print_status(cur_t, len(key_list), caption, gt_labels, pred_labels, hand_eval_dict[my_key], other_hand_eval_val=other_hand_eval_val)
        s = input()
        if s in ['w', 'q']:
            print('writing output to file "%s"'%(hand_eval_dict_filename))
            with open(hand_eval_dict_filename, 'wb') as f:
                pickle.dump(hand_eval_dict, f)

            if s == 'q':
                print('quitting!')
                break
            else:
                assert(s == 'w')
        elif s == 'n':
            cur_t = (cur_t + 1) % len(key_list)
        elif s == 'p':
            cur_t = (cur_t - 1) % len(key_list)
        elif any([s.startswith(p) for p in ['correct:', 'mild:', 'severe:', 'erase correct:', 'erase mild:', 'erase severe:']]):
            if s.startswith('erase'):
                add_or_erase = 'erase'
            else:
                add_or_erase = 'add'

            if s.startswith('correct:') or s.startswith('erase correct:'):
                hand_type = 'correct'
            elif s.startswith('mild:') or s.startswith('erase mild:'):
                hand_type = 'mild'
            elif s.startswith('severe:') or s.startswith('erase severe:'):
                hand_type = 'severe'
            else:
                assert(False)

            payload = s[int(add_or_erase == 'erase') * len('erase ') + len(hand_type + ':'):]
            if payload[0] != '"' or payload[-1] != '"':
                print('badly formatted payload, try again')
                continue
            pair = payload[1:-1].split('","')
            if len(pair) != 2:
                print('badly formatted payload, try again')
                continue
            pair = (sanitize_label(pair[0]), sanitize_label(pair[1]))
            gt_labels = set(sanitize_labels(gt_labels))
            pred_labels = set(sanitize_labels(pred_labels))
            if add_or_erase == 'add':
                if pair[0] not in gt_labels - pred_labels:
                    print('first thing in pair is not in gt_labels - pred_labels, please try again')
                    continue
                if pair[1] not in pred_labels - gt_labels:
                    print('second thing in pair is not in pred_labels - gt_labels, please try again')
                    continue
                if any([pair[0] in [p[0] for p in sorted(hand_eval_dict[my_key][some_hand_type])] for some_hand_type in ['correct','mild','severe']]):
                    print('pair[0] is already in "correct", "mild", or "severe". cannot have multiple matches to it. need to erase something.')
                    continue
                if any([pair[1] in [p[1] for p in sorted(hand_eval_dict[my_key][some_hand_type])] for some_hand_type in ['correct','mild','severe']]):
                    print('pair[1] is already in "correct", "mild", or "severe". cannot have multiple matches to it. need to erase something.')
                    continue
                hand_eval_dict[my_key][hand_type].add(pair)
            elif add_or_erase == 'erase':
                if pair not in hand_eval_dict[my_key][hand_type]:
                    print('pair not in hand_eval_dict[my_key][hand_type], cannot erase')
                    continue
                assert(pair[0] in gt_labels - pred_labels)
                assert(pair[1] in pred_labels - gt_labels)
                hand_eval_dict[my_key][hand_type].remove(pair)
            else:
                assert(False)

def usage():
    print('Usage: python hand_eval.py <gt_label_dict_filename> <pred_label_dict_filename> <hand_eval_dict_filename> <random_seed> [<other_hand_eval_dict_filename>=None]')

if __name__ == '__main__':
    hand_eval(*(sys.argv[1:]))
