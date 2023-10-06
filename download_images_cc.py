import os
import sys
import glob
import pandas as pd
import pickle
from img2dataset import download


IMAGE_SIZE = 256
DEFAULT_PROCESS_COUNT = 12
ENCODE_FORMAT = 'jpg'
ENCODE_QUALITY = 95


def download_images_cc(cc_csv_filename, image_dir, process_count=DEFAULT_PROCESS_COUNT):
    process_count = int(process_count)

    url_filename = os.path.join(image_dir, 'image_urls.txt')
    os.makedirs(image_dir, exist_ok=True)
    df = pd.read_csv(cc_csv_filename, header=None)
    f = open(url_filename, 'w')
    for url in df[2]:
        f.write(url + '\n')

    f.close()

    with open(os.path.join(image_dir, 'image_bases.pkl'), 'wb') as f:
        pickle.dump(df[0], f)

    output_dir = os.path.join(image_dir, 'downloaded_images')
    os.makedirs(output_dir, exist_ok=True)
#    download(image_size=IMAGE_SIZE, processes_count=process_count, url_list=url_filename, output_folder=output_dir, output_format='files', input_format='txt', resize_mode='keep_ratio', encode_format=ENCODE_FORMAT, encode_quality=ENCODE_QUALITY)


def usage():
    print('Usage: python download_images_cc.py <cc_csv_filename> <image_dir> [<process_count>=%d]'%(DEFAULT_PROCESS_COUNT))


if __name__ == '__main__':
    download_images_cc(*(sys.argv[1:]))
