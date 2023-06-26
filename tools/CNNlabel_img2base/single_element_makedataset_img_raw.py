import os
import glob
import numpy as np
import cv2
from miji.imageProcessNorm import imgnorm as norm99
from scipy.ndimage import grey_dilation, generate_binary_structure

"""

"""

def norm(arr):  # 对每个通道各自做归一化。
    for i in range(arr.shape[0]):
        channel = arr[i]
        min_val = np.amin(channel)
        max_val = np.amax(channel)
        arr[i] = (channel - min_val) / (max_val - min_val)
    return arr

def gaussReplace(matrix,base_type):
    ones_matrix = np.where(matrix == base_type, 1, 0).astype(np.float64)
    return ones_matrix
"""
file's name rule: img/machine_fov_cycle_img.npy    
                  label/machine_fov_cycle_label.npy
                  msk/machine_fov_msk.npy
                  img/
                            
"""
save_root_dir = r"C:\deepdata\image"
readRootDir = r"E:\data\resize_test\08_resize_ori\res_deep"
machine_name = '08h'
fov = 'R001C001'
paths = glob.glob(fr"{readRootDir}\Image\Lane01\*\{fov}_A.jpg")
path_labels = glob.glob(fr"{readRootDir}\Lane01\deepLearnData\*\label\{fov}_label.npy")
mask_path =rf"{readRootDir}\Lane01\deepLearnData\{fov}_mask.npy"
msk = np.load(mask_path) #
for (path,path_label) in zip(paths,path_labels) :

    print("oriname:", path_label)
    #for name in namelist:
    idx = 1

    name_path_C = path.replace("_A", '_C')
    name_path_G = path.replace("_A", '_G')
    name_path_T = path.replace("_A", '_T')


    imgA = cv2.imread(path,0)
    imgC = cv2.imread(name_path_C, 0)
    imgG = cv2.imread(name_path_G, 0)
    imgT = cv2.imread(name_path_T, 0)


    label = np.load(path_label)

    cycname = path.split("\\")[-2]
    # os.makedirs(f"{val}/{machine_name}_{fov}_imgdata", exist_ok=True)
    # os.makedirs(f"{val}/{machine_name}_{fov}_label", exist_ok=True)
    cv2.imwrite(f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_A_img.tif", imgA)
    cv2.imwrite(f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_C_img.tif", imgC)
    cv2.imwrite(f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_G_img.tif", imgG)
    cv2.imwrite(f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_T_img.tif", imgT)

    np.save(f"{save_root_dir}/label/{machine_name}_{fov}_{cycname}_label.npy", label)
    idx = idx + 1
    print("idx：", idx)
np.save(f"{save_root_dir}/msk/{machine_name}_{fov}_msk.npy",msk)



