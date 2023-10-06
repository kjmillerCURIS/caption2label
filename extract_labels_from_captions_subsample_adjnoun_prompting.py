import os
import sys
import math
import pandas as pd
import pickle
import random
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from hand_label_captions import load_captions
from hand_eval import sanitize_label, sanitize_labels, is_adj_noun

LLAMA_PATH_DICT = {'7B' : '/usr3/graduate/nivek/data/vislang-domain-exploration-data/caption2label-data/llama-models/llama-7b-hf',
                    '13B' : '/usr3/graduate/nivek/data/vislang-domain-exploration-data/caption2label-data/llama-models/llama-13b'}
MIN_MAX_NEW_TOKENS = 32
PROMPT_CONTEXT = 'Pick the class names from these captions:'
BATCH_SIZE = 8
RANDOM_SEED = 0
TOP_K = None
TOP_P = 0.95

#in case I don't trust qsub
def write_to_log_file(msg):
    print(msg)
    f = open('meow.txt', 'a')
    f.write(msg + '\n')
    f.close()

def load_llama(llama_type):
    llama_path = LLAMA_PATH_DICT[llama_type]
    model = LlamaForCausalLM.from_pretrained(llama_path, torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained(llama_path, torch_dtype=torch.float16, padding_side='left')
    tokenizer.pad_token = '<eos>' #I have no idea why this works, but it results in a pad token of 0, and generation_config.json says it should be 0
    model = model.cuda()
    model.eval()
    return model, tokenizer

#will return labelsA, labelsB
#where labelsA has full adj_nouns and labelsB has only nouns
def split_labels(labels):
    cur_i = 0
    labelsA = []
    labelsB = []
    while cur_i < len(labels):
        if cur_i < len(labels) - 1 and is_adj_noun(sanitize_label(labels[cur_i]), sanitize_label(labels[cur_i + 1])):
            labelsA.append(labels[cur_i])
            labelsB.append(labels[cur_i + 1])
            cur_i += 2
        else:
            labelsA.append(labels[cur_i])
            labelsB.append(labels[cur_i])
            cur_i += 1

    return labelsA, labelsB

#yes, this does do random shuffling, so the ordering of the support samples is randomized
#returns prompt_pair
#it also returns caption_token_len, which is the number of tokens used when tokenizing just the caption
def make_one_prompt_pair(caption, metalabel_dict, tokenizer):
    caption_token_len = tokenizer(caption, return_tensors='pt')['input_ids'].shape[1]
    key_list = sorted(metalabel_dict.keys())
    random.shuffle(key_list) 
    supportsA = []
    supportsB = []
    for k in key_list:
        labelsA, labelsB = split_labels(metalabel_dict[k]['labels'])
        supportsA.append('%s=>'%(metalabel_dict[k]['caption']) + ','.join(['%s'%(label) for label in labelsA]))
        supportsB.append('%s=>'%(metalabel_dict[k]['caption']) + ','.join(['%s'%(label) for label in labelsB]))

    promptA = PROMPT_CONTEXT + '\n' + '\n'.join(supportsA) + '\n' + '%s=>'%(caption)
    promptB = PROMPT_CONTEXT + '\n' + '\n'.join(supportsB) + '\n' + '%s=>'%(caption)
    prompt_pair = (promptA, promptB)
    return prompt_pair, caption_token_len

#use ALL 16 entries in metalabel_dict, in random order
#returns prompt_pairs, max_new_tokens
#max_new_tokens is guaranteed to be at least MIN_MAX_NEW_TOKENS
def make_prompt_pairs(captions, metalabel_dict, tokenizer):
    prompt_pairs = []
    max_new_tokens = MIN_MAX_NEW_TOKENS
    for caption in captions:
        prompt_pair, caption_token_len = make_one_prompt_pair(caption, metalabel_dict, tokenizer)
        prompt_pairs.append(prompt_pair)
        max_new_tokens = max(2*caption_token_len, max_new_tokens)

    return prompt_pairs, max_new_tokens

def postprocess(output, prompt):
    assert(prompt == output[:len(prompt)])
    labels = output[len(prompt):].split('\n')[0].split(',')
    return labels

#only let through stuff from labelsB if it's a suffix of something from labelsA
#return filtered labelsB
def filter_noun_only_labels(labelsA, labelsB):
    labelsB_filtered = []
    labelsA_san = sanitize_labels(labelsA)
    for labelB in labelsB:
        labelB_san = sanitize_label(labelB)
        if labelB_san == '':
            continue

        for labelA_san in labelsA_san:
            if is_adj_noun(labelA_san, labelB_san):
                labelsB_filtered.append(labelB)
                break

    return labelsB_filtered

#return labels_listA, labels_listB, labels_listB_prefilter
def process_batch(captions, metalabel_dict, tokenizer, model, temperature):
#        write_to_log_file('about to make_prompt_pairs')
        prompt_pairs, max_new_tokens = make_prompt_pairs(captions, metalabel_dict, tokenizer)
#        write_to_log_file('done make_prompt_pairs. about to tokenizer')
        inputsA = tokenizer([p[0] for p in prompt_pairs], return_tensors='pt', padding=True)
        inputsB = tokenizer([p[1] for p in prompt_pairs], return_tensors='pt', padding=True)
#        write_to_log_file('done tokenizer')
        inputsA['input_ids'] = inputsA['input_ids'].cuda()
        inputsB['input_ids'] = inputsB['input_ids'].cuda()
        inputsA['attention_mask'] = inputsA['attention_mask'].cuda()
        inputsB['attention_mask'] = inputsB['attention_mask'].cuda()
        with torch.no_grad():
            #FIXME: Figure out if we need to set do_sample=True to make things like temperature meaningful (I don't think so)
#            write_to_log_file('about to model.generate')
            generate_idsA = model.generate(**inputsA, max_new_tokens=max_new_tokens, temperature=temperature, top_k=TOP_K, top_p=TOP_P)
            generate_idsB = model.generate(**inputsB, max_new_tokens=max_new_tokens, temperature=temperature, top_k=TOP_K, top_p=TOP_P)
#            write_to_log_file('done model.generate, about to tokenizer.batch_decode')
            outputsA = tokenizer.batch_decode(generate_idsA, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            outputsB = tokenizer.batch_decode(generate_idsB, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#            write_to_log_file('done tokenizer.batch_decode')

#        write_to_log_file('about to postprocess and dict')
        labels_listA = [postprocess(output, prompt_pair[0]) for output, prompt_pair in zip(outputsA, prompt_pairs)]
        labels_listB_prefilter = [postprocess(output, prompt_pair[1]) for output, prompt_pair in zip(outputsB, prompt_pairs)]
        labels_listB = [filter_noun_only_labels(labelsA, labelsB) for labelsA, labelsB in zip(labels_listA, labels_listB_prefilter)]
        return labels_listA, labels_listB, labels_listB_prefilter

def extract_labels_from_captions_subsample_adjnoun_prompting(caption_csv_filename, caption_key_list_filename, metalabel_dict_filename, llama_type, temperature, output_dict_filename):
    temperature = float(temperature)

    with open(caption_key_list_filename, 'rb') as f:
        caption_key_list = pickle.load(f)

    write_to_log_file('loading captions...')
    caption_dict = load_captions(caption_csv_filename)
    write_to_log_file('done loading captions')

    write_to_log_file('loading supports...')
    with open(metalabel_dict_filename, 'rb') as f:
        metalabel_dict = pickle.load(f)

    assert(not any([k in metalabel_dict for k in caption_key_list]))
    write_to_log_file('done loading supports')

    model, tokenizer = load_llama(llama_type)

    random.seed(RANDOM_SEED)

    output_dict = {}
    cur_t = 0
    for _ in tqdm(range(int(math.ceil(len(caption_key_list) / BATCH_SIZE)))):
        if cur_t >= len(caption_key_list):
            break

        caption_key_batch = caption_key_list[cur_t:min(cur_t + BATCH_SIZE, len(caption_key_list))]
        captions = [caption_dict[k] for k in caption_key_batch]
        labels_listA, labels_listB, labels_listB_prefilter = process_batch(captions, metalabel_dict, tokenizer, model, temperature)
        for k, caption, labelsA, labelsB, labelsB_prefilter in zip(caption_key_batch, captions, labels_listA, labels_listB, labels_listB_prefilter):
            output_dict[k] = {'caption' : caption, 'labels' : labelsA + labelsB, 'labels_structured' : {'full_adj_noun' : labelsA, 'noun_only_prefilter' : labelsB_prefilter, 'noun_only' : labelsB}}

#        write_to_log_file('done postprocess and dict')

        cur_t += BATCH_SIZE

    write_to_log_file('about to dump')
    with open(output_dict_filename, 'wb') as f:
        pickle.dump(output_dict, f)

    write_to_log_file('done dump')

def usage():
    print('Usage: python extract_labels_from_captions_subsample_adjnoun_prompting.py <caption_csv_filename> <caption_key_list_filename> <metalabel_dict_filename> <llama_type> <temperature> <output_dict_filename>')

if __name__ == '__main__':
    extract_labels_from_captions_subsample_adjnoun_prompting(*(sys.argv[1:]))
