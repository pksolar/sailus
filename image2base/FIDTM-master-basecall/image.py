import scipy.spatial
from PIL import Image
import scipy.io as io
import scipy
import numpy as np
import h5py
import cv2

def norm(arr):
    for i in range(arr.shape[0]):
        channel = arr[i]
        min_val = np.amin(channel)
        max_val = np.amax(channel)
        arr[i] = (channel - min_val) / (max_val - min_val)
    return arr


def load_data_fidt_test(img_path, args, train=True):


    while True:
        try:
            img_sheng = np.load(img_path)
            img = norm(img_sheng)
            break
        except OSError:
            print("path is wrong, can not load ", img_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()

    return img

def load_data_fidt(img_path, args, train=True):
    gt_path = img_path.replace('imgdata', 'label')
    mask_path_sring = img_path.replace('imgdata','mask')
    mask_path = mask_path_sring[:-15]+mask_path_sring[-8:]



    while True:
        try:
            img_sheng = np.load(img_path)
            img = norm(img_sheng)
            gt = np.load(gt_path)
            msk = np.load(mask_path)
            break
        except OSError:
            print("path is wrong, can not load ", img_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    gt = gt.copy()
    msk = msk.copy()


    return img, gt,msk
