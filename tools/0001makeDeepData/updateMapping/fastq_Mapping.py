import os
import glob
import numpy as np
import cv2
import shutil


def autoMap(fastq_dir,fastq_name="Lane01_fastq",mappingmode= "very-sensitive-local"): #fastq所在文件夹，fastq的name
    #run_version = fr"{machine_name}{key}"
    # dir_path = fr'{ori_mappingdir}\Lane01\\'
    fastq_file = fr'{fastq_dir}\\{fastq_name}.fq'
    gz_file = fr'{fastq_dir}\\{fastq_name}.fq.gz'
    sam_file = fr'{fastq_dir}\\{fastq_name}.fq.sam'

    stater_file = fr'{fastq_dir}\\{fastq_name}.fq_total.out'
    split_file = fr'{fastq_dir}\\{fastq_name}.fq_split.out'


    index_file = r'E:\software\sailu\WinMappingScript\reference\Ecoli.fa_index'
    # 第一条命令
    # cmd1 = fr'E:\software\sailu\WinMappingScript\software\7z.exe x {gz_file} -y -o{ori_mappingdir}'
    # os.system(cmd1)

    # 第二条命令
    cmd2 = fr'E:\software\sailu\WinMappingScript\software\bowtie2-align-s.exe --{mappingmode} -p 10 -q {fastq_file} -S {sam_file} -x {index_file}'
    os.system(cmd2)

    # 第三条命令
    cmd3 = fr'python E:\software\sailu\WinMappingScript\samStaterThree.py {sam_file} {stater_file} --splitFov {split_file}'

    os.system(cmd3)



if __name__=="__main__":
    #fastq文件所在文件夹
    ori_mappingdir = r"E:\code\python_PK\img2base_preprocess\fastq\44.1h_98.299_0620144328\Lane01"

    #fastq名字
    fastq_name =  "Lane01_fastq" #

    #mapping的模式：local: --sensitive-local,--very-sensitive-local，--very-fast-local,--fast-local，end2end：--very-fast，--very-sensitive，--fast，
    mappingmode = "--very-fast"
    autoMap(ori_mappingdir,fastq_name=fastq_name,mappingmode=mappingmode)