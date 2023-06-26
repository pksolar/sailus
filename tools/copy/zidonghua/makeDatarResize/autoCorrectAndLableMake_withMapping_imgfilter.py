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
machineName = "55.2h_resize_ori"
rootdir = rf"E:\data\resize_test\{machineName}" # 下一级目录是FOV
fovlsit = ["R001C001"]
save_root_dir = r"E:\data\testAuto"
for fov in fovlsit:
    print(fov)
    #machineName = fov
    rootdir = fr"{rootdir}\\"
    resdir = rootdir+"res_deep_imageFilter"
    ori_mappingdir = resdir

    # 步骤1，运行baseCall，为了生成用来mapping的文件fastq以及生成配准后的图像。
    step1_command = rf'E:\software\DeepLearningDataMake\001SalusCall_imageFilter.exe {rootdir} {resdir} 1 100 1 1 -m'
    os.system(step1_command)
    # print("baseCall  done")
    # #插入mapping 步骤：
    # autoMap(ori_mappingdir)
    # print("Mapping done")
    # # 步骤2，校正
    # step2_command = rf'python E:\CrosstalkTool\fastqCorrect.py {ori_mappingdir}\Lane01\Lane01_fastq.fq.sam {ori_mappingdir}\Lane01\Lane01_fastq.fq.gz {ori_mappingdir}\Lane01\samResult'
    # os.system(step2_command)
    # print("correct done")
    # # 步骤3，生成label
    # step3_command = rf'python E:\CrosstalkTool\generData_PengKuanKuan_resize.py {ori_mappingdir}\Lane01\sfile {ori_mappingdir}\Lane01\sfile {ori_mappingdir}\Lane01\samResult {resdir}\Lane01\deepLearnData'
    # os.system(step3_command)
    # print("label done")
    # # 步骤4
    # step4_command = r'echo Step 4 is complete'
    # os.system(step4_command)

    #autoCopy(resdir,fov,machineName,save_root_dir)

#然后再用这个文件夹下的copy将copy到c盘。

