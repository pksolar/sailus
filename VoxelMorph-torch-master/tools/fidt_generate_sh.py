import glob
import math
import os
import torch
import cv2
import h5py
import numpy as np
# import scipy.io as io
# import scipy.spatial
# from scipy.ndimage.filters import gaussian_filter
def fidt_generate1(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)

    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda * gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    distance_map = torch.from_numpy(distance_map)
    distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0

    return distance_map
'''change your path'''
#
# 4张图：
typelist = ['A','C','G','T']


for type_ in typelist:
    root = rf'E:\data\resize_test\17_R1C78_resize_ori\Lane01\Cyc001\R001C001_{type_}.tif'
    print( rf'E:\data\resize_test\17_R1C78_resize_ori\Lane01\Cyc001\R001C001_{type_}.tif')
    Img_data = cv2.imread(root)

    Gt_data = np.loadtxt(rf"E:\data\resize_test\17_R1C78_resize_ori\res_for_reg\Lane01\sfile\R001C001_chanel_{type_}.btemp",skiprows=2) #

    fidt_map1 = fidt_generate1(Img_data, Gt_data, 1)

    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for i in range(0, len(Gt_data)):
        if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1

    fidt_map1 = fidt_map1
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    # fidt_map1 = cv2.applyColorMap(fidt_map1, 2)

    '''for visualization'''
    cv2.imwrite(fr'{type_}_fidt.png', fidt_map1)
