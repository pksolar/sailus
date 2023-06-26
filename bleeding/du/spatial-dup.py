#!/usr/bin/env python3

#################################################################################
## Copyright: Salus-bio Corp 2022
## Author: Huihua Xia
## Date of creation: 6/15/2022
#
## Project: Quality estimation of sequencing data
## Description
## - Project: duplication
#
## - Goal: This script is to count the duplicated reads of all reads of input fastq file.
#
#################################################################################

import os
import gzip
import pandas as pd
import argparse
import  numpy as np
import cv2
import glob

def getOutlier(coordDirTotal,name, gamma=1.2):
    if gamma == '_ori':
        gamma = 1
    threshDis = gamma * 1 # threshDis 是计算这个距离内的点的个数
    ### get Outlier for each FOV
    peakT = coordDirTotal #peakT 是list。转化为nx2的array
    peak_array = np.zeros((len(peakT),2))
    for i ,content in enumerate(peakT):
        x, y = map(lambda x: float(x),content.split())
        peak_array[i][0] = x
        peak_array[i][1] = y

    #map(lambda x,y :peak_array[i]=np.array(content.split(" ") for i, content in enumerate(peakT))

    peakT = peak_array
    peak = peakT.T
    nearReadSet = []  ### reacord Outlier readId
    for readId, peakTemp in enumerate(peakT):#peakTemp 是坐标，x,y
        #### calculate each
        nearPos = np.where((peak[0] < peakTemp[0] + threshDis) & (peak[0] > peakTemp[0] - threshDis) &
                 (peak[1] < peakTemp[1] + threshDis) & (peak[1] > peakTemp[1] - threshDis))[0]
        if len(nearPos) == 1:  #这个距离内的点只有一个的话，认为只有它自己。
            nearReadSet.append([readId, peakTemp[0], peakTemp[1]])
            continue
        else: #否则，就是说还有别的点。，计算一下这个点，到中心点的距离。
            difX = np.square(peak[0][nearPos] - peakTemp[0])
            difY = np.square(peak[1][nearPos] - peakTemp[1])
            disTemp = np.sqrt(difX + difY)  # np.sqrt(difX + difY)

            nearNum = len(np.where(disTemp < threshDis)[0])

            if nearNum == 1:
                #print(np.sort(disTemp))
                nearReadSet.append([readId, peakTemp[0], peakTemp[1]])
    #### out nearRead Id to file
    #print("nearReadSet num:", nearReadSet)
    print(rf"data: {name},the near dup num is :{len(coordDirTotal)-len(nearReadSet)}")
    return len(nearReadSet)

    #np.savetxt(os.path.join(outPath, FOV + OulierSuffix), np.array(nearReadSet))






"""
Count numbers of duplicated reads.

Parmas:
  odir: output dir for results of duplication.
  sample_id: sample id for the output file.
  fqgz: input fastq.gz file.
  n_num: input int, 0 indicates do not filter sequences containing N,
    1/2/3/... indicates filter to retain sequences containing < 1/2/3/... N.

Return: None
    """
    # "1.9.1_resize", "1.9.39_resize",
    # odir
machineNames = ["30_resize", "22_resize", "08_resize",
                "17_R1C78_resize"]  # "1.9.1_resize","1.9.39_resize",

dict_resize = {"_ori": [2160, 4096],
               1.15: [2484, 4710],
               1.2: [2592, 4914],
               1.25: [2700, 5120],
               1.3: [2808, 4324],
               1.35: [2916, 5530],
               1.4: [3024, 5734]}
odir  = r'E:\code\python_PK\bleeding\du'

    # remove N ?
readId_list = []
for machine_name in machineNames:
    for key ,value in dict_resize.items():
        try:
            seq_f = fr"E:\data\resize_test\{machine_name}{key}\res\Lane01\Lane01_fastq.fq"
            sample_id = rf'{machine_name}{key}_dup'
            # count read numbers of each unique read
            seq_counts = {}
            total_read_num = 0
            with open(seq_f) as fh:
                    #print(rf"read{machine_name}{key} successfully")
                    for i, line in enumerate(fh):
                        if i % 4 == 1:
                            total_read_num += 1
                            seq = line.strip()
                            seq_counts[seq] = seq_counts.get(seq, 0) + 1
                            if seq_counts[seq] > 1: #说明这个seq是复制的seq，那么找到它的id
                                readId = total_read_num
                                readId_list.append(readId) #保存了所有的id。根据temp，从图中标记出来。保存成图像。
        except:
            print(f"error{key}")
            continue


        # 读取txt文件
        with open(rf'E:\data\resize_test\{machine_name}{key}\res\Lane01\sfile\R001C001.temp') as f:
            lines = f.readlines()[2:]
        # 生成一个黑底图
        img = np.zeros((dict_resize[key][0], dict_resize[key][1], 3), np.uint8)

        # 设置坐标点颜色为绿色
        color = (0, 255, 0)
        #得到所有dup reads 的坐标：
        dup_corr = [lines[index_i-1] for index_i in readId_list]
        #采用孤立点计算方法，看其中多少孤立点。8邻域内没有别的点就是孤立点。但是需要计算出缩小到4096到2160内的点的新坐标。采用点插值的方式。采用矩阵并行计算方式：
        name = rf"{machine_name}{key}"
        num_near_dup = getOutlier(dup_corr,name,gamma=key)


        # 读取list中每一行的坐标信息并画在黑底图上
        for line_num in readId_list:
            try:
                x, y = map(lambda x : int(float(x)), lines[line_num - 1].split())
                #将x,y保存到新的
                cv2.circle(img, (x, y), 1, color, -1)
            except:
                print("line_num:",line_num)
                print(len(lines))
                print(machine_name,key)


        # 展示图像
        cv2.imwrite(fr"{machine_name}{key}R1C1.jpg",img)
        readId_list = []

        #对图进行分析，有多少个点在8邻域内出现。
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        #cv2.destroyAllWindows()



        # extract duplicated reads
        dupseq_counts = {}
        dup_read_num = dup_read_nums = 0
        for k, v in seq_counts.items():
            if v > 1:
                dup_read_num += 1
                dup_read_nums += v
                dupseq_counts[k] = v

        # duplication rate
        # total_read_num = total_read_num / 4
        dup_rate1 = dup_read_num / total_read_num * 100
        dup_rate2 = dup_read_nums / total_read_num * 100
        dup_rate_res = f"{odir}/{sample_id}.dup_rate.txt"
        os.system(f'echo "Sample_id: {sample_id}" > {dup_rate_res}')
        os.system(f'echo "Total_reads: {total_read_num}" >> {dup_rate_res}')
        os.system(f'echo "Unique_duplicated_reads: {dup_read_num}" >> {dup_rate_res}')
        os.system(f'echo "All_duplicated_reads: {dup_read_nums}" >> {dup_rate_res}')
        os.system(f'echo "Unique_duplication_rate(%): {dup_rate1}" >> {dup_rate_res}')
        os.system(f'echo "All_duplication_rate(%): {dup_rate2}" >> {dup_rate_res}')

        # save duplicated reads and their read numbers
        df = pd.DataFrame.from_dict(dupseq_counts, orient="index")
        df.columns = ["read_numbers"]
        df["read_numbers/all_dup_read_numbers(%)"] = (
            df["read_numbers"] / dup_read_nums * 100
        )
        ## keep 3 significant digits
        new_ratio = []
        for i in range(len(df)):
            new_r = float("{:.3}".format(df.iloc[i, 1]))
            new_ratio.append(new_r)
        df['read_numbers/all_dup_read_numbers(%)'] = new_ratio
        ## sort and save
        df.index.name = "read_seq"
        df = df.sort_values(["read_numbers"], ascending=False)
        dup_res = f"{odir}/{sample_id}.dup_read_numbers.txt"
        df.to_csv(dup_res, sep="\t")

