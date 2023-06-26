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

    filterPath = instensityPath.replace("intent","imageFilter")
    noFilterPath =instensityPath.replace("intent","No_imageFilter")
    machine_name = machinename[:2]+'h'

    filterPath_imgs = glob.glob(fr"{filterPath}\Image\Lane01\*\{fov}_A.jpg")
    noFilterPath_imgs = glob.glob(fr"{noFilterPath}\Image1\Lane01\*\{fov}_A.jpg")


    path_labels = glob.glob(fr"{instensityPath}\Lane01\deepLearnData\*\label\{fov}_label.npy")
    mask_path = rf"{instensityPath}\Lane01\deepLearnData\{fov}_mask.npy"
    os.makedirs(f"{save_root_dir}/msk", exist_ok=True)
    shutil.copy2(mask_path, f"{save_root_dir}/msk/{machine_name}_{fov}_msk.npy")

    for (filterPath_img, noFilterPath_img,path_label) in zip(filterPath_imgs, noFilterPath_imgs,path_labels):
        print("oriname:", path_label)
        # for name in namelist:
        idx = 1

        filterPath_path_C = filterPath_img.replace("_A", '_C')
        filterPath_path_G = filterPath_img.replace("_A", '_G')
        filterPath_path_T = filterPath_img.replace("_A", '_T')

        nofilterPath_path_C = noFilterPath_img.replace("_A", '_C')
        nofilterPath_path_G = noFilterPath_img.replace("_A", '_G')
        nofilterPath_path_T = noFilterPath_img.replace("_A", '_T')

        cycname = filterPath_img.split("\\")[-2]

        os.makedirs(f"{save_root_dir}/img", exist_ok=True)
        os.makedirs(f"{save_root_dir}/img_ori", exist_ok=True)
        os.makedirs(f"{save_root_dir}/label", exist_ok=True)

        shutil.copy2(filterPath_img, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_A.tif")
        shutil.copy2(filterPath_path_C, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_C.tif")
        shutil.copy2(filterPath_path_G, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_G.tif")
        shutil.copy2(filterPath_path_T, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_T.tif")

        shutil.copy2(noFilterPath_img, f"{save_root_dir}/img_ori/{machine_name}_{fov}_{cycname}_A.tif")
        shutil.copy2(nofilterPath_path_C, f"{save_root_dir}/img_ori/{machine_name}_{fov}_{cycname}_C.tif")
        shutil.copy2(nofilterPath_path_G, f"{save_root_dir}/img_ori/{machine_name}_{fov}_{cycname}_G.tif")
        shutil.copy2(nofilterPath_path_T, f"{save_root_dir}/img_ori/{machine_name}_{fov}_{cycname}_T.tif")



        shutil.copy2(path_label, f"{save_root_dir}/label/{machine_name}_{fov}_{cycname}.npy")

        idx = idx + 1
        print("idx：", idx)
if __name__ == "__main__":
   print("ddd")