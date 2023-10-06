import os
import sys
import pickle

def prettify_extractions(extraction_dict_filename, pretty_filename):
    with open(extraction_dict_filename, 'rb') as f:
        extraction_dict = pickle.load(f)

    f = open(pretty_filename, 'w')
    for k in sorted(extraction_dict.keys()):
        f.write(extraction_dict[k]['caption'] + '\t' + ','.join(extraction_dict[k]['labels']) + '\n')

    f.close()

def usage():
    print('Usage: python prettify_extractions.py <extraction_dict_filename> <pretty_filename>')

if __name__ == '__main__':
    prettify_extractions(*(sys.argv[1:]))
