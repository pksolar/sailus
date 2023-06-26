import os
import time
import glob
import cv2
import h5py
import numpy as np
import scipy.io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import math
import scipy.io as io
from matplotlib import pyplot as plt
import sys

'''please set your dataset path'''
root = 'E:\code\python_PK\image2base\FIDTM-master-base\data/noname/test_data/'

part_A_test = os.path.join(root, 'images')
path_sets = [part_A_test]

if not os.path.exists(part_A_test):
    sys.exit("The path is wrong, please check the dataset path.")


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()

f = open('point_files/base_gt.txt', 'w+')
k = 1
for img_path in img_paths:

    print(img_path)
    Gt_data = np.load(img_path.replace('.jpg', '.npy').replace('images', 'posnpy').replace('img', 'array'))
    f.write('{} {} '.format(k, len(Gt_data)))

    for data in Gt_data:

        sigma_s = 4
        sigma_l = 8
        f.write('{} {} {} {} {} '.format(math.floor(data[0]), math.floor(data[1]), sigma_s, sigma_l, 1))
    f.write('\n')

    k = k + 1
f.close()
