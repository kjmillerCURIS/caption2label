import os
import sys
import pickle
from tqdm import tqdm
from write_to_log_file import write_to_log_file


def map_image_filenames(image_dir):
    image_filename_dict = {}
    with open(os.path.join(image_dir, 'image_bases.pkl'), 'rb') as f:
        image_key_list = pickle.load(f)

    write_to_log_file('there are %d image_bases'%(len(image_key_list)))
    for i, k in tqdm(enumerate(image_key_list)):
        if i % 1000 == 0:
            write_to_log_file(str(i))

        s = '%09d'%(i)
        image_filename = os.path.join(image_dir, 'downloaded_images', s[:5], s + '.jpg')
        if os.path.exists(image_filename):
            image_filename_dict[k] = image_filename

    write_to_log_file('found %d images'%(len(image_filename_dict)))
    with open(os.path.join(image_dir, 'image_filename_dict.pkl'), 'wb') as f:
        pickle.dump(image_filename_dict, f)


def usage():
    print('Usage: python map_image_filenames.py <image_dir>')


if __name__ == '__main__':
    map_image_filenames(*(sys.argv[1:]))
