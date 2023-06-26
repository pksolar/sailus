import numpy as np
import os
import glob
# 所有cycle里的亮度矩阵路径
total_dir = glob.glob("E:\code\python_PK\callbase\datasets/21/Res/Lane01/deepLearnData/*/intensity/*.npy")
def imgnorm(img,p1,p99):
    """
    采用什么归一化方式：
    1、除255
    2、0-255
    :param image:
    :return:
    normalize the input images
    """
    norm = (img-p1)/(p99-p1)
    return norm
def dellist(list_i,value):
    list_out = []
    for i in list_i:
        if i != 0:
            list_out.append(i)
    return list_out
def cal99_1(img):
    """
      :param img: array 矩阵
      :return:
      """
    a, b = img.shape
    # 将矩阵拉成条
    array_2 = img.reshape(1, a * b)[0, :]
    # 变成列表
    array_list = array_2.tolist()
    # 删除0
    array_list2 = dellist(array_list, 0)
    # 升序排列
    array_list_sorted = sorted(array_list2)  # 升序排列
    list_len = len(array_list_sorted)
    p1_idx = int(0.01 * list_len)
    p99_idx = int(0.99 * list_len)
    p1 = array_list_sorted[p1_idx]
    p99 = array_list_sorted[p99_idx]
    return p1,p99

for path in total_dir:
    img = np.load(path)
    p1, p99 = cal99_1(img)
    img = imgnorm(img, p1, p99)
    # 用相同的名字保存在不同的文件夹里。
    path_norm = path.replace("intensity","intensity_norm")
    path_norm_dir = path_norm[:-14]
    if not os.path.exists(path_norm_dir):
         os.makedirs(path_norm_dir)
    np.save(path_norm,img)
    print(path[-30:])
    # if


