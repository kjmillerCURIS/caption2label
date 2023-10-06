import os
import sys
import pickle
from tqdm import tqdm
from extract_labels_from_captions_subsample_adjnoun_prompting import load_llama, process_batch
from hand_label_captions import load_captions

LLAMA_TYPE = '13B'
TEMPERATURE = 0.7
BATCH_SIZE = 8
SAVE_FREQ = 100 #in batches

#return list of lists of keys
#first we use start_index and stride to figure out what keys we're responsible for processing
#then we skip any keys that have already been processed
#then we put the remainder of the keys into batches
def get_key_batches(caption_dict, start_index, stride, output_dict):
    my_keys = sorted(caption_dict.keys())
    my_keys = [k for i, k in enumerate(my_keys) if i % stride == start_index % stride]
    my_keys = [k for k in my_keys if k not in output_dict]
    key_batches = []
    cur_batch = []
    for k in my_keys:
        cur_batch.append(k)
        if len(cur_batch) == BATCH_SIZE:
            key_batches.append(cur_batch)
            cur_batch = []
        else:
            assert(len(cur_batch) < BATCH_SIZE)

    if len(cur_batch) > 0:
        key_batches.append(cur_batch)

    return key_batches

def extract_labels_from_captions_bulk(caption_csv_filename, metalabel_dict_filename, start_index, stride, output_dict_filename):
    start_index = int(start_index)
    stride = int(stride)

    print('loading captions...')
    caption_dict = load_captions(caption_csv_filename)
    print('done loading captions')

    print('loading metalabel_dict...')
    with open(metalabel_dict_filename, 'rb') as f:
        metalabel_dict = pickle.load(f)
    print('done loading metalabel_dict')

    print('loading llama...')
    model, tokenizer = load_llama(LLAMA_TYPE)
    print('done loading llama')

    print('loading/creating output_dict...')
    output_dict = {}
    if os.path.exists(output_dict_filename):
        with open(output_dict_filename, 'rb') as f:
            output_dict = pickle.load(f)

    print('done loading/creating output_dict')

    print('getting key_batches...')
    key_batches = get_key_batches(caption_dict, start_index, stride, output_dict)
    print('done getting key_batches')

    save_counter = 0
    for key_batch in tqdm(key_batches):
        captions = [caption_dict[k] for k in key_batch]
        labels_listA, labels_listB, labels_listB_prefilter = process_batch(captions, metalabel_dict, tokenizer, model, TEMPERATURE)
        for k, labelsA, labelsB, labelsB_prefilter in zip(key_batch, labels_listA, labels_listB, labels_listB_prefilter):
            output_dict[k] = {'labels' : labelsA + labelsB, 'labels_structured' : {'full_adj_noun' : labelsA, 'noun_only_prefilter' : labelsB_prefilter, 'noun_only' : labelsB}}
        save_counter += 1
        if save_counter >= SAVE_FREQ:
            print('saving...')
            with open(output_dict_filename, 'wb') as f:
                pickle.dump(output_dict, f)

            print('done saving')
            save_counter = 0

    print('final saving...')
    with open(output_dict_filename, 'wb') as f:
        pickle.dump(output_dict, f)

    print('done final saving')

def usage():
    print('Usage: python extract_labels_from_captions_bulk.py <caption_csv_filename> <metalabel_dict_filename> <start_index> <stride> <output_filename>')

if __name__ == '__main__':
    extract_labels_from_captions_bulk(*(sys.argv[1:]))
