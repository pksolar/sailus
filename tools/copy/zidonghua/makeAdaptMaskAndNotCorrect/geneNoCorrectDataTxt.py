import numpy as np
"""
    现在的是对call出来的碱基进行过校正，然后得到一个label
    
    我需要对call出来的错误的碱基，不校正，并且当场生成一个mask对这个错误的碱基进行覆盖。
    
    我将会生成未校正的label和校正后的label进行比较，然后不同的地方就生成mask，进行覆盖。
    
    msk的区域只用和原来的mask同时作用，就能让不该计算loss的地方不计算loss
    
    
    1、将q30后的fastq文件写成txt形式，并重命名为ACGTcorrect
    3、跑datadgenda.
        
    首先这些数据都是有fastq文件的。
"""

# 读取fastq并转化：
machine_name = "17_R1C78_resize_ori"
rootpath = rf"E:\data\resize_test\{machine_name}\res_deep_intent\Lane01\\"
filename = "Lane01_fastq.fq"
with open(rootpath+filename,'r') as f:
    content = f.readlines()
    # idx =  range(1:328478 * 4,4)
    result = content[1::4]
with open(rootpath+ "samResult//"+ rf"R001C001correctACGT_False.txt",'w') as f:
    for item in result:
        f.write(item)
#



