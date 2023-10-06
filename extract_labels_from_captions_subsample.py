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

def load_llama(model_path):
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained(model_path, torch_dtype=torch.float16, padding_side='left')
    tokenizer.pad_token = '<eos>' #I have no idea why this works, but it results in a pad token of 0, and generation_config.json says it should be 0
    model = model.cuda()
    model.eval()
    return model, tokenizer

#yes, this does do random shuffling, so the ordering of the support samples is randomized
#it also returns caption_token_len, which is the number of tokens used when tokenizing just the caption
def make_one_prompt(caption, metalabel_dict, tokenizer):
    caption_token_len = tokenizer(caption, return_tensors='pt')['input_ids'].shape[1]
    key_list = sorted(metalabel_dict.keys())
    random.shuffle(key_list)
    supports = ['%s=>'%(metalabel_dict[k]['caption']) + ','.join(['%s'%(label) for label in metalabel_dict[k]['labels']]) for k in key_list]
    prompt = PROMPT_CONTEXT + '\n' + '\n'.join(supports) + '\n' + '%s=>'%(caption)
    return prompt, caption_token_len

#use ALL 16 entries in metalabel_dict, in random order
#returns prompts, max_new_tokens
#max_new_tokens is guaranteed to be at least MIN_MAX_NEW_TOKENS
def make_prompts(captions, metalabel_dict, tokenizer):
    prompts = []
    max_new_tokens = MIN_MAX_NEW_TOKENS
    for caption in captions:
        prompt, caption_token_len = make_one_prompt(caption, metalabel_dict, tokenizer)
        prompts.append(prompt)
        max_new_tokens = max(2*caption_token_len, max_new_tokens)

    return prompts, max_new_tokens

def postprocess(output, prompt):
    assert(prompt == output[:len(prompt)])
    labels = output[len(prompt):].split('\n')[0].split(',')
    return labels

def extract_labels_from_captions_subsample(caption_csv_filename, caption_key_list_filename, metalabel_dict_filename, llama_type, temperature, output_dict_filename):
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

    write_to_log_file('getting llama_path')
    llama_path = LLAMA_PATH_DICT[llama_type]
    write_to_log_file('loading llama...')
    model, tokenizer = load_llama(llama_path)
    write_to_log_file('done loading llama')

    random.seed(RANDOM_SEED)

    output_dict = {}
    cur_t = 0
    for _ in tqdm(range(int(math.ceil(len(caption_key_list) / BATCH_SIZE)))):
        if cur_t >= len(caption_key_list):
            break

        caption_key_batch = caption_key_list[cur_t:min(cur_t + BATCH_SIZE, len(caption_key_list))]
        captions = [caption_dict[k] for k in caption_key_batch]
        write_to_log_file('about to make_prompts')
        prompts, max_new_tokens = make_prompts(captions, metalabel_dict, tokenizer)
        write_to_log_file('done make_prompts. about to tokenizer')
        inputs = tokenizer(prompts, return_tensors='pt', padding=True)
        write_to_log_file('done tokenizer')
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        with torch.no_grad():
            #FIXME: Figure out if we need to set do_sample=True to make things like temperature meaningful (I don't think so)
            write_to_log_file('about to model.generate')
            generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_k=TOP_K, top_p=TOP_P)
            write_to_log_file('done model.generate, about to tokenizer.batch_decode')
            outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            write_to_log_file('done tokenizer.batch_decode')

        write_to_log_file('about to postprocess and dict')
        labels_list = [postprocess(output, prompt) for output, prompt in zip(outputs, prompts)]
        for k, caption, labels in zip(caption_key_batch, captions, labels_list):
            output_dict[k] = {'caption' : caption, 'labels' : labels}

        write_to_log_file('done postprocess and dict')

        cur_t += BATCH_SIZE

    write_to_log_file('about to dump')
    with open(output_dict_filename, 'wb') as f:
        pickle.dump(output_dict, f)

    write_to_log_file('done dump')

def usage():
    print('Usage: python extract_labels_from_captions_subsample.py <caption_csv_filename> <caption_key_list_filename> <metalabel_dict_filename> <llama_type> <temperature> <output_dict_filename>')

if __name__ == '__main__':
    extract_labels_from_captions_subsample(*(sys.argv[1:]))
