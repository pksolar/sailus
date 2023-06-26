import os
import glob
import numpy as np
import cv2
import shutil
import subprocess





def autoMap(ori_mappingdir):
    #run_version = fr"{machine_name}{key}"
    dir_path = fr'{ori_mappingdir}\Lane01\\'
    fastq_file = fr'{ori_mappingdir}\Lane01\Lane01_test.fastq'
    gz_file = fr'{ori_mappingdir}\Lane01\Lane01_test.fastq.gz'
    sam_file = fr'{ori_mappingdir}\Lane01\Lane01_test.fastq.sam'

    stater_file = fr'{ori_mappingdir}\\Lane01\Lane01_fastq.fq_total.out'
    split_file = fr'{ori_mappingdir}\\Lane01\Lane01_fastq.fq_split.out'

    index_file = r'E:\software\sailu\WinMappingScript\reference\Ecoli.fa_index'
    # 第一条命令
    cmd1 = fr'E:\software\sailu\WinMappingScript\software\7z.exe x {gz_file} -y -o{dir_path}'
    os.system(cmd1)

    # 第二条命令
    cmd2 = fr'E:\software\sailu\WinMappingScript\software\bowtie2-align-s.exe --very-fast -p 10 -q {fastq_file} -S {sam_file} -x {index_file}'
    os.system(cmd2)
    process = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    output, error = process.communicate()

    try:
        readsNum = float(error.split(" ")[0])
        result = error.split(" ")[-4][-6:-1]
        result = float(result)
        return result, readsNum * result / 100
    except:
        print("!!!!may be can not detect any clusters!!!")

        result = 0
        return 0, 0


    # # 第三条命令
    # cmd3 = fr'python E:\software\sailu\WinMappingScript\samStaterThree.py {sam_file} {stater_file} --splitFov {split_file}'
    #
    # os.system(cmd3)
# machineName = "R001C001"
def main(rootdir,resdirname,cycleNum = 30):
    # rootdir = r"E:\code\python_PK\bleeding\img_process\scaleDownPro\Gauss08_-0.1-0.2_2.5\\" # 下一级目录是FOV
    # resdirname = "res4.24cyc50"
    # cycleNum = 50

    fovlsit = ["R001C001"]
    for fov in fovlsit:

        resdir = rootdir + resdirname
        # ori_mappingdir = fr"{rootdir}\\res_deep"

        # # 步骤1，运行baseCall，为了生成用来mapping的文件fastq以及生成配准后的图像。
        step1_command = rf"E:\software\sailu5.25_highDense\001SalusCall.exe {rootdir} {resdir} 1 {cycleNum} 1 1 -m"
        os.system(step1_command)
        print("baseCall  done")
        #插入mapping 步骤：
        mapping,mappedReads = autoMap(resdir)
        print("Mapping done")
        return mapping,mappedReads

if __name__ == "__main__":
    main(r"E:\code\python_PK\bleeding\img_process\scaleDownPro\Gauss08_-0.1-0.2_2.5\\", "res4.24cyc50")
#然后再用这个文件夹下的copy将copy到c盘。

