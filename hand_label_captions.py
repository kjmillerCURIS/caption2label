import os
import sys
import pandas as pd
import pickle

def load_captions(caption_csv_filename):
    df = pd.read_csv(caption_csv_filename, header=None)
    return {k : v for k, v in zip(df[0], df[1])}

def print_status(caption_dict, metalabel_dict, my_key, key_list):
    print('')
    print('%d/%d filled'%(len(metalabel_dict), len(key_list)))
    print('Key: %s'%(my_key))
    print('Caption: "%s"'%(caption_dict[my_key]))
    if my_key in metalabel_dict:
        assert(caption_dict[my_key] == metalabel_dict[my_key]['caption'])
        if len(metalabel_dict[my_key]['labels']) == 0:
            print('Labels: EMPTY')
        else:
            print('Labels: %s'%(','.join(['"%s"'%(label) for label in metalabel_dict[my_key]['labels']])))

def hand_label_captions(caption_csv_filename, key_list_filename, metalabel_dict_filename):
    print('loading captions...')
    caption_dict = load_captions(caption_csv_filename)
    print('done loading captions')

    with open(key_list_filename, 'rb') as f:
        key_list = pickle.load(f)

    metalabel_dict = {}
    if os.path.exists(metalabel_dict_filename):
        with open(metalabel_dict_filename, 'rb') as f:
            metalabel_dict = pickle.load(f)

    cur_t = 0
    while True:
        my_key = key_list[cur_t]
        print_status(caption_dict, metalabel_dict, my_key, key_list)
        while True:
            s = input()
            if len(s) == 0:
                continue

            if s in ['EMPTY', 'n', 'p', 'w', 'q']:
                break

            if s[0] == '"' and s[-1] == '"':
                break

        if s in ['w', 'q']: #save output
            print('writing output to file "%s"'%(metalabel_dict_filename))
            with open(metalabel_dict_filename, 'wb') as f:
                pickle.dump(metalabel_dict, f)

            if s == 'w': #keep going
                continue
            if s == 'q': #quit
                print('do svidaniya!')
                break

        if s == 'n': #advance without updating
            cur_t = (cur_t + 1) % len(key_list)
        elif s == 'p': #prev without updating
            cur_t = (cur_t - 1) % len(key_list)
        elif s == 'EMPTY': #update with empty list, and advance
            metalabel_dict[my_key] = {'caption' : caption_dict[my_key], 'labels' : []}
            cur_t = (cur_t + 1) % len(key_list)
        else:
            assert(s[0] == '"' and s[-1] == '"')
            labels = s[1:-1].split('","')
            assert(len(labels) >= 1)
            metalabel_dict[my_key] = {'caption' : caption_dict[my_key], 'labels' : labels}
            cur_t = (cur_t + 1) % len(key_list)

def usage():
    print('Usage: python hand_label_captions.py <caption_csv_filename> <key_list_filename> <metalabel_dict_filename>')

if __name__ == '__main__':
    hand_label_captions(*(sys.argv[1:]))
