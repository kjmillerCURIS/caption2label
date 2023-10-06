import os
import sys
import glob
from PIL import Image
import pickle
from tqdm import tqdm

#will assume "ILSVRC2012_val" dir and "classnames.txt" file are already in dataset_parent_dir/ImageNet1K
#will write "dataset.pkl" file in same directory with the following:
#-"image_partialpath_list" has list of *valid* partial paths for everything after "ILSVRC2012_val"
#-"gt_list" is parallel list of gts (which are ordered by the folder in alphabetical order)
#-"classnames" is list of classnames ordered by gt, completely contiguous
def setup_imagenet1k_eval_dataset(dataset_parent_dir):
    folder2classname = {}
    f = open(os.path.join(dataset_parent_dir, 'ImageNet1K', 'classnames.txt'), 'r')
    for line in f:
        ss = line.rstrip('\n').split(' ')
        folder = ss[0]
        classname = ' '.join(ss[1:])
        assert(folder not in folder2classname)
        folder2classname[folder] = classname

    f.close()
    sorted_folders = sorted(folder2classname.keys())
    assert(len(sorted_folders) == 1000)
    classnames = [folder2classname[folder] for folder in sorted_folders]

    image_partialpath_list = []
    gt_list = []
    assert(len(glob.glob(os.path.join(dataset_parent_dir, 'ImageNet1K', 'ILSVRC2012_val', '*'))) == 1000)
    for gt, folder in tqdm(enumerate(sorted_folders)):
        images = sorted(glob.glob(os.path.join(dataset_parent_dir, 'ImageNet1K', 'ILSVRC2012_val', folder, '*.JPEG')))
        assert(len(images) > 0)
        for image in images:
            if not os.path.exists(image):
                print('skip %s'%(image))

            print('about to open %s'%(image))
            Image.open(image)
            image_partialpath = os.path.join(folder, os.path.basename(image))
            assert(len(image_partialpath_list) == len(gt_list))
            image_partialpath_list.append(image_partialpath)
            gt_list.append(gt)
            assert(len(image_partialpath_list) == len(gt_list))

    ds = {'classnames' : classnames, 'image_partialpath_list' : image_partialpath_list, 'gt_list' : gt_list}
    with open(os.path.join(dataset_parent_dir, 'ImageNet1K', 'dataset.pkl'), 'wb') as f:
        pickle.dump(ds, f)


def usage():
    print('Usage: python setup_imagenet1k_eval_dataset.py <dataset_parent_dir>')


if __name__ == '__main__':
    setup_imagenet1k_eval_dataset(*(sys.argv[1:]))
