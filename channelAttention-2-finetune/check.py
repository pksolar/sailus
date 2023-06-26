import torch.utils.data as data
from datasets_ import Dataset_npy, Dataset_npy_val
import os
from natsort import natsorted
import numpy as np
from mymodel import DNA_Sequencer_Atten
import fastQ
import time
import numpy as np

new = np.load(r"E:\code\python_PK\channelAttention-2-finetune\fastq\08h_R001C001_99.029_20230607-171956\Lane01\deepLearnDataUpdate\R001C001_mask.npy")
old = np.load(r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\R001C001_mask.npy")
label_old =np.load(r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\Cyc001\label\R001C001_label.npy")
label_new = np.load(r"E:\code\python_PK\channelAttention-2-finetune\fastq\08h_R001C001_99.029_20230607-171956\Lane01\deepLearnDataUpdate\Cyc001\label\R001C001_label.npy")
label_bing =  np.load(r"E:\code\python_PK\channelAttention-2-finetune\fastq\bingjilabel\Lane01\deepLearnDataUpdate\Cyc001\label\R001C001_label.npy")
g = np.where(label_bing != label_new)
print("old",label_old[3,1985])
print("new",label_new[3,1985])
print("bing",label_bing[3,1985])
print(np.array_equal(label_bing,label_new))
print("ddd")
def poscheck():
    # 两个输入矩阵
    A = np.array([[1, 0, -1], [0, 1, 0], [1, -1, 0]])
    B = np.array([[0, 1, 0], [-1, 0, 1], [0, 0, -1]])

    # 判断两个矩阵中是否存在1
    contains_one = np.logical_or(A == 1, B == 1)

    # 生成新的矩阵
    result = np.where(contains_one, 1, A)

    print("新矩阵:")
    print(result)



def labelcheck():
    oldlabel =r"E:\data\liangdujuzhen\label\08h_R001C001_withNoMap.npy"
    newlabel = r"E:\data\liangdujuzhen\update\label\08h_R001C001_withNoMap.npy"
    new = np.load(newlabel)
    old = np.load(oldlabel)
    print("xxx")

def makscheck():
    new = np.load(r"E:\code\python_PK\channelAttention-2-finetune\fastq\08h_R001C001_99.029_20230607-171956\Lane01\deepLearnDataUpdate\R001C001_mask.npy")
    old = np.load(r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\R001C001_mask.npy")
    c = new - old
    a = np.where(c == 2)
    b = np.where(c == -2)

    print("ddd")

#
# def autoMap(ori_mappingdir):
#     #run_version = fr"{machine_name}{key}"
#     dir_path = fr'{ori_mappingdir}\\'
#     fastq_file = fr'{ori_mappingdir}\\Lane01_fastq.fq'
#     gz_file = fr'{ori_mappingdir}\\Lane01_fastq.fq.gz'
#     sam_file = fr'{ori_mappingdir}\\Lane01_fastq.fq.sam'
#
#     stater_file = fr'{ori_mappingdir}\\Lane01_fastq.fq_total.out'
#     split_file = fr'{ori_mappingdir}\\Lane01_fastq.fq_split.out'
#
#
#     index_file = r'E:\software\sailu\WinMappingScript\reference\Ecoli.fa_index'
#     # # 第一条命令
#     # cmd1 = fr'E:\software\sailu\WinMappingScript\software\7z.exe x {gz_file} -y -o{dir_path}'
#     # os.system(cmd1)
#
#     # 第二条命令
#     cmd2 = fr'E:\software\sailu\WinMappingScript\software\bowtie2-align-s.exe --very-fast -p 10 -q {fastq_file} -S {sam_file} -x {index_file}'
#     os.system(cmd2)
#
#     # 第三条命令
#     cmd3 = fr'python E:\software\sailu\WinMappingScript\samStaterThree.py {sam_file} {stater_file} --splitFov {split_file}'
#
#     os.system(cmd3)
# dir =r"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\44h_R001C001_97.742_0605173808\Lane01"
# autoMap(dir)