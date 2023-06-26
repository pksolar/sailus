import os
import glob
import numpy as np
import cv2
import shutil
from miji.imageProcessNorm import imgnorm as norm99
from scipy.ndimage import grey_dilation, generate_binary_structure



"""
file's name rule: img/machine_fov_cycle_img.npy    
                  label/machine_fov_cycle_label.npy
                  msk/machine_fov_msk.npy
                  img/
                            
"""
save_root_dir = r"E:\data\testAuto"
fov = 'R001C001'
readRootDir = rf"E:\data\testAuto\dense0.6\{fov}\res_deep"
machine_name = '144h'

paths = glob.glob(fr"{readRootDir}\Image\Lane01\*\{fov}_A.jpg")
path_labels = glob.glob(fr"{readRootDir}\Lane01\deepLearnData\*\label\{fov}_label.npy")
mask_path =rf"{readRootDir}\Lane01\deepLearnData\{fov}_mask.npy"
os.makedirs(f"{save_root_dir}/msk", exist_ok=True)
shutil.copy2(mask_path, f"{save_root_dir}/msk/{machine_name}_{fov}_msk.npy")

for (path,path_label) in zip(paths,path_labels) :

    print("oriname:", path_label)
    #for name in namelist:
    idx = 1

    name_path_C = path.replace("_A", '_C')
    name_path_G = path.replace("_A", '_G')
    name_path_T = path.replace("_A", '_T')

    cycname = path.split("\\")[-2]

    os.makedirs(f"{save_root_dir}/img",exist_ok=True)
    os.makedirs(f"{save_root_dir}/label", exist_ok=True)


    shutil.copy2(path,f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_A_img.tif")
    shutil.copy2(name_path_C, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_C_img.tif")
    shutil.copy2(name_path_G, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_G_img.tif")
    shutil.copy2(name_path_T, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_T_img.tif")

    shutil.copy2(path_label, f"{save_root_dir}/label/{machine_name}_{fov}_{cycname}_label.npy")

    idx = idx + 1
    print("idx：", idx)




