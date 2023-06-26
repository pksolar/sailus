import os
import glob
import numpy as np
import cv2
import shutil
def autoCopy(instensityPath,fov,machinename,save_root_dir):
    """
    file's name rule:
    从3个文件夹来读取训练用的图像。是用分开的文件夹还是不分开？
    分开，末尾不做区分。
    """
    # save_root_dir = r"E:\data\testAuto"
    # fov = 'R001C001'

    machine_name = machinename[:2]+'h'

    # path_labels = glob.glob(fr"{instensityPath}\Lane01\deepLearnData\*\label\{fov}_label.npy")
    adapt_masks = glob.glob(fr"{instensityPath}\Lane01\deepLearnData\*\msk\{fov}_msk.npy")
    for adapt_mask in adapt_masks:
        print("oriname:", adapt_mask)
        # for name in namelist:
        idx = 1

        cycname = adapt_mask.split("\\")[-3]

        os.makedirs(f"{save_root_dir}/mskAdapt", exist_ok=True)

        shutil.copy2(adapt_mask, f"{save_root_dir}/mskAdapt/{machine_name}_{fov}_{cycname}.npy")

        idx = idx + 1
        print("idx：", idx)
if __name__ == "__main__":
    machineName = r"17_resize_ori"
    autoCopy(rf"E:\data\resize_test\{machineName}\res_deep_intent\\","R001C001",machineName,r"E:\data\testAuto\\")