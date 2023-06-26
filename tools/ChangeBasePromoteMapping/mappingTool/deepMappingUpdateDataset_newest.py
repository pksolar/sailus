import os
import glob
import numpy as np
import cv2
import shutil
from argparse import ArgumentParser
import re
from fastq_Mapping import autoMap

from datetime import datetime

def autoCopy(fastq_dir,machinename,fov,save_root_dir,):
    """
    file's name rule: img/machine_fov_cycle_img.npy
                      label/machine_fov_cycle_label.npy
                      msk/machine_fov_msk.npy
                      img/

    """
    # save_root_dir = r"E:\data\testAuto"
    # fov = 'R001C001'


    # paths = glob.glob(fr"{readRootDir}\Image\Lane01\*\{fov}_A.jpg")
    path_labels = glob.glob(fr"{fastq_dir}\\deepLearnData\*\label\{fov}_label.npy")
    mask_path = rf"{fastq_dir}\\deepLearnData\{fov}_mask.npy"
    os.makedirs(f"{save_root_dir}/msk", exist_ok=True)
    shutil.copy2(mask_path, f"{save_root_dir}/msk/{machinename}_{fov}_msk.npy")

    for path_label in path_labels:
        # print("oriname:", path_label)
        # for name in namelist:
        idx = 1

        # name_path_C = path.replace("_A", '_C')
        # name_path_G = path.replace("_A", '_G')
        # name_path_T = path.replace("_A", '_T')

        cycname = path_label.split("\\")[-3]

        os.makedirs(f"{save_root_dir}/img", exist_ok=True)
        os.makedirs(f"{save_root_dir}/label", exist_ok=True)

        # shutil.copy2(path, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_A_img.tif")
        # shutil.copy2(name_path_C, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_C_img.tif")
        # shutil.copy2(name_path_G, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_G_img.tif")
        # shutil.copy2(name_path_T, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_T_img.tif")

        shutil.copy2(path_label, f"{save_root_dir}/label/{machinename}_{fov}_{cycname}.npy")
        idx = idx + 1
        print("idx：", idx)



def deepUpdateData(fastq_dir,machinename,save_root_dir,mask_path,fastq_name="Lane01_fastq",mappingmode = "very-sensitive-local",mapping=True,correct=True,genlabel = True,autocopy=True):
    """
    save_root_dir: fastq res 所在目录
    machineName： 机器编号 如 08,44h等等
    fov
    save_root_dir 生成的新的数据集放置的位置，如果存在则覆盖。


    """
    with open(os.path.join(fastq_dir,"mappinglog.txt"),"w") as f:
        f.write("fastdir: "+fastq_dir+"\n")
        f.write("deep dir: "+ save_root_dir + "\n")
        f.write("mask dir:"+mask_path + "\n")
        f.write("fastq name: "+fastq_name + "\n")
        f.write("mapping mode: "+ mappingmode + "\n")
        f.write("if mapping："+ str(mapping) + "\n")
        f.write("if correct：" +str(correct) + "\n")
        f.write("if genlabel：" + str(genlabel) + "\n")
        f.write("if autocopy：" +str(autocopy) + "\n")
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        f.write("time：" + formatted_time + "\n")

    # machineName = "R001C001"
    #rootdir = r"E:\data\testAuto\dense0.6" # 下一级目录是FOV
    # fovlsit = ["R001C001"]
    # save_root_dir = r"E:\data\testAuto"

    # # 插入mapping 步骤，这里是座mapping的步骤。
    if mapping:
        autoMap(fastq_dir,fastq_name=fastq_name,mappingmode=mappingmode)
    # # 步骤2，校正
    if correct:
        step2_command = rf'python E:\CrosstalkTool\fastqCorrect.py {fastq_dir}\\{fastq_name}.fq.sam {fastq_dir}\\{fastq_name}.fq {fastq_dir}\\samResult'
        os.system(step2_command)
        print("step2  done")
    #步骤3,生成label
    #sfile_path = rf'E:\code\python_PK\img2base_cnn_seg\fastq\144h_R001C001_97.049_0509140318\Lane01\sfile' #用传统方法生成的 sfile_path
    if genlabel:
        step3_command = rf'python E:\CrosstalkTool\generData_PengKuanKuan_deep.py  {mask_path} {fastq_dir}\\samResult {fastq_dir}\\deepLearnData'
        os.system(step3_command)
        print("step3  done")
    # 步骤4
    step4_command = r'echo Step 4 is complete'
    os.system(step4_command)
    # fov = fastq_name.split("_")[1]
    #fov去fastq_name找R_C_找到后读取fov，没有找到就默认为：R001C00

    pattern = r"R[0-9]{3}C[0-9]{3}"  # 匹配 R 后面跟着三位数字，然后是 C 后面跟着三位数字的字符串

    matches = re.findall(pattern, fastq_dir)  # 找到所有匹配的字符串
    if len(matches) > 0 :
        fov = matches[0]
    else:
        fov = "R001C001"
    if autocopy:
        autoCopy(fastq_dir,machinename,fov,save_root_dir) #
        print("copyDone")



#def deepUpdateData(rootdir,machineName,fov,time,save_root_dir):
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fd",
                        type=str,
                        dest="fastq_dir",
                        default=r"E:\data\fastqResult\bingji95.09_very_sensitive_local\label_fastq\label_fastq")

    parser.add_argument("--mp", #用最原始的mask即可，只是为了获取坐标
                        type=str,
                        dest="mask_path",
                        default = r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\R001C001_mask.npy")

    parser.add_argument("--mn",
                        type=str,
                        dest="machinename",
                        default = r"08.2hvslvfvsl")

    parser.add_argument("--fn",
                        type=str,
                        dest="fastq_name",
                        default = "Lane01_fastq") #不以.fq结尾

    parser.add_argument("--dd",
                        type=str,
                        dest="deep_dir",
                        default=r"E:\data\testAuto_test")
    parser.add_argument("--mm",
                        type=str,
                        dest="mappingmode",
                        default=r"very-fast")

    opt = parser.parse_args()


    #fast文件所在的文件夹
    fastq_dir = opt.fastq_dir

    #file_name = "change_R001C001_sensitive_local"


    #os.makedirs(rootdir + rf"//{file_name}", exist_ok=True)

    #用来生成label以及mask时，需要老的mask的坐标信息，注意，fastq文件必须是从左到右，从上到下生成的。
    mask_path =opt.mask_path

    #用来标记深度学习数据
    machinename = opt.machinename
    fastq_name = opt.fastq_name
    mappingmode = opt.mappingmode

    # 将制作的数据集copy保存到深度学习用的数据集下
    deep_dir = opt.deep_dir
    os.makedirs(deep_dir, exist_ok=True)

    deepUpdateData(fastq_dir,machinename,deep_dir,mask_path,mappingmode=mappingmode,
                   fastq_name = fastq_name,
                   mapping=True,correct=True,genlabel=True,autocopy=True)
