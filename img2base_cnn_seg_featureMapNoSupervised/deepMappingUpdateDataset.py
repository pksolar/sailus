import os
import glob
import numpy as np
import cv2
import shutil


def autoCopy(rootdir,fov,save_root_dir):
    """
    file's name rule: img/machine_fov_cycle_img.npy
                      label/machine_fov_cycle_label.npy
                      msk/machine_fov_msk.npy
                      img/

    """
    # save_root_dir = r"E:\data\testAuto"
    # fov = 'R001C001'
    readRootDir = rootdir
    machine_name = '08hub'

    # paths = glob.glob(fr"{readRootDir}\Image\Lane01\*\{fov}_A.jpg")
    path_labels = glob.glob(fr"{readRootDir}\Lane01\deepLearnData\*\label\{fov}_label.npy")
    mask_path = rf"{readRootDir}\Lane01\deepLearnData\{fov}_mask.npy"
    os.makedirs(f"{save_root_dir}/msk", exist_ok=True)
    shutil.copy2(mask_path, f"{save_root_dir}/msk/{machine_name}_{fov}_msk.npy")

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

        shutil.copy2(path_label, f"{save_root_dir}/label/{machine_name}_{fov}_{cycname}.npy")

        idx = idx + 1
        print("idx：", idx)


def autoMap(ori_mappingdir,fastq_name): #fastq所在文件夹，fastq的name
    #run_version = fr"{machine_name}{key}"
    # dir_path = fr'{ori_mappingdir}\Lane01\\'
    fastq_file = fr'{ori_mappingdir}\Lane01\Lane01_fastq.fq'
    gz_file = fr'{ori_mappingdir}\Lane01\Lane01_fastq.fq.gz'
    sam_file = fr'{ori_mappingdir}\Lane01\Lane01_fastq.fq.sam'

    stater_file = fr'{ori_mappingdir}\Lane01\Lane01_fastq.fq_total.out'
    split_file = fr'{ori_mappingdir}\Lane01\Lane01_fastq.fq_split.out'


    index_file = r'E:\software\sailu\WinMappingScript\reference\Ecoli.fa_index'
    # 第一条命令
    # cmd1 = fr'E:\software\sailu\WinMappingScript\software\7z.exe x {gz_file} -y -o{ori_mappingdir}'
    # os.system(cmd1)

    # 第二条命令
    cmd2 = fr'E:\software\sailu\WinMappingScript\software\bowtie2-align-s.exe --very-fast -p 10 -q {fastq_file} -S {sam_file} -x {index_file}'
    os.system(cmd2)

    # 第三条命令
    cmd3 = fr'python E:\software\sailu\WinMappingScript\samStaterThree.py {sam_file} {stater_file} --splitFov {split_file}'

    os.system(cmd3)



def deepUpdateData(rootdir,fastq_name,save_root_dir,mask_path,mapping=True,correct=True,genlabel = True,autocopy=True):
    """
    save_root_dir: fastq res 所在目录
    machineName： 机器编号 如 08,44h等等
    fov
    save_root_dir 生成的新的数据集放置的位置，如果存在则覆盖。


    """
    # machineName = "R001C001"
    #rootdir = r"E:\data\testAuto\dense0.6" # 下一级目录是FOV
    # fovlsit = ["R001C001"]
    # save_root_dir = r"E:\data\testAuto"

    resdir = rootdir + rf"//{fastq_name}"
    ori_mappingdir = resdir
    # # 插入mapping 步骤，这里是座mapping的步骤。
    if mapping:
        autoMap(ori_mappingdir,fastq_name=fastq_name)
    # # 步骤2，校正
    if correct:
        step2_command = rf'python E:\CrosstalkTool\fastqCorrect.py {ori_mappingdir}\Lane01\Lane01_fastq.fq.sam {ori_mappingdir}\Lane01\Lane01_fastq.fq {ori_mappingdir}\Lane01\samResult'
        os.system(step2_command)
        print("step2  done")
    #步骤3,生成label
    #sfile_path = rf'E:\code\python_PK\img2base_cnn_seg\fastq\144h_R001C001_97.049_0509140318\Lane01\sfile' #用传统方法生成的 sfile_path
    if genlabel:
        step3_command = rf'python E:\CrosstalkTool\generData_PengKuanKuan_deep.py  {mask_path} {ori_mappingdir}\Lane01\samResult {resdir}\Lane01\deepLearnData'
        os.system(step3_command)
        print("step3  done")
    # 步骤4
    step4_command = r'echo Step 4 is complete'
    os.system(step4_command)
    fov = fastq_name.split("_")[1]
    if autocopy:
        autoCopy(resdir,fov,save_root_dir) #
        print("copyDone")

#def deepUpdateData(rootdir,machineName,fov,time,save_root_dir):
if __name__ == "__main__":
    rootdir = r"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq"
    #验证集会给出machine，fov，acc，time,不必我去读。
    file_name = "08h_R001C001_98.404_0613121439"
    save_root_dir = r"E:\data\testAuto"
    os.makedirs(rootdir + rf"//{file_name}", exist_ok=True)
    mask_path = r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\R001C001_mask.npy"
    deepUpdateData(rootdir,file_name,save_root_dir,mask_path,mapping=False,correct=False,genlabel=False,autocopy=True)
