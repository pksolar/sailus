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
    readRootDir = rf"{rootdir}\\res_deep"
    machine_name = '144h'

    paths = glob.glob(fr"{readRootDir}\Image\Lane01\*\{fov}_A.jpg")
    path_labels = glob.glob(fr"{readRootDir}\Lane01\deepLearnData\*\label\{fov}_label.npy")
    mask_path = rf"{readRootDir}\Lane01\deepLearnData\{fov}_mask.npy"
    os.makedirs(f"{save_root_dir}/msk", exist_ok=True)
    shutil.copy2(mask_path, f"{save_root_dir}/msk/{machine_name}_{fov}_msk.npy")

    for (path, path_label) in zip(paths, path_labels):
        print("oriname:", path_label)
        # for name in namelist:
        idx = 1

        name_path_C = path.replace("_A", '_C')
        name_path_G = path.replace("_A", '_G')
        name_path_T = path.replace("_A", '_T')

        cycname = path.split("\\")[-2]

        os.makedirs(f"{save_root_dir}/img", exist_ok=True)
        os.makedirs(f"{save_root_dir}/label", exist_ok=True)

        shutil.copy2(path, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_A_img.tif")
        shutil.copy2(name_path_C, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_C_img.tif")
        shutil.copy2(name_path_G, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_G_img.tif")
        shutil.copy2(name_path_T, f"{save_root_dir}/img/{machine_name}_{fov}_{cycname}_T_img.tif")

        shutil.copy2(path_label, f"{save_root_dir}/label/{machine_name}_{fov}_{cycname}_label.npy")

        idx = idx + 1
        print("idx：", idx)
def autoMap(ori_mappingdir):
    #run_version = fr"{machine_name}{key}"
    dir_path = fr'{ori_mappingdir}\Lane01\\'
    fastq_file = fr'{ori_mappingdir}\Lane01\Lane01_fastq.fq'
    gz_file = fr'{ori_mappingdir}\Lane01\Lane01_fastq.fq.gz'
    sam_file = fr'{ori_mappingdir}\Lane01\Lane01_fastq.fq.sam'

    stater_file = fr'{ori_mappingdir}\\Lane01\Lane01_fastq.fq_total.out'
    split_file = fr'{ori_mappingdir}\\Lane01\Lane01_fastq.fq_split.out'


    index_file = r'E:\software\sailu\WinMappingScript\reference\Ecoli.fa_index'
    # 第一条命令
    cmd1 = fr'E:\software\sailu\WinMappingScript\software\7z.exe x {gz_file} -y -o{dir_path}'
    os.system(cmd1)

    # 第二条命令
    cmd2 = fr'E:\software\sailu\WinMappingScript\software\bowtie2-align-s.exe --very-fast -p 10 -q {fastq_file} -S {sam_file} -x {index_file}'
    os.system(cmd2)

    # 第三条命令
    cmd3 = fr'python E:\software\sailu\WinMappingScript\samStaterThree.py {sam_file} {stater_file} --splitFov {split_file}'

    os.system(cmd3)
# machineName = "R001C001"
rootdir = r"E:\data\resize_test\08.2h_resize_ori" # 下一级目录是FOV
fovlsit = ["R001C001"]
save_root_dir = r"E:\data\testAuto"
for fov in fovlsit:
    print(fov)
    machineName = fov
    #rootdir = fr"{rootdir}\{fov}\\"
    #resdir = rootdir+"res_deep"
    ori_mappingdir = fr"{rootdir}\\res_deep_intent"

    # # 步骤1，运行baseCall，为了生成用来mapping的文件fastq以及生成配准后的图像。
    # step1_command = rf'E:\software\sailu\SalusCall_deeplearnTrue.exe {rootdir} {resdir} 1 10 1 1 -m'
    # os.system(step1_command)
    # print("baseCall  done")
    #插入mapping 步骤：
    autoMap(ori_mappingdir)
    print("Mapping done")
    # 步骤2，校正
    step2_command = rf'python E:\CrosstalkTool\fastqCorrect.py {ori_mappingdir}\Lane01\Lane01_fastq.fq.sam {ori_mappingdir}\Lane01\Lane01_fastq.fq.gz {ori_mappingdir}\Lane01\samResult'
    os.system(step2_command)
    print("correct done")
    # 步骤3，生成label
    step3_command = rf'python E:\CrosstalkTool\generData_PengKuanKuan.py {ori_mappingdir}\Lane01\sfile {ori_mappingdir}\Lane01\sfile {ori_mappingdir}\Lane01\samResult {ori_mappingdir}\Lane01\deepLearnData'
    os.system(step3_command)
    print("label done")
    # 步骤4
    step4_command = r'echo Step 4 is complete'
    os.system(step4_command)
    #autoCopy(rootdir,fov,save_root_dir)

#然后再用这个文件夹下的copy将copy到c盘。

