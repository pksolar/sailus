import Levenshtein
import time
import numpy as np
# 定义碱基序列片段长度
length = 100
dict_acgt = {'A':1,"C":2,"G":3,"T":4,"N":0}
# 定义阈值n
thresh = 5
total_read_num = 0
# 读入碱基序列片段保存在txt文件中
sequences = []
sequences_num = []
list_a = []
seq_f = r"E:\code\python_PK\bleeding\img_process\22_resize1.35\res\Lane01\Lane01_fastq.fq"
with open(seq_f, 'r') as f:
    for i,line in enumerate(f) :
        if i % 4 == 1:
            line = line.strip()
            total_read_num += 1
            sequences.append(line)

for seq in sequences:
    for j in seq:
        list_a.append(dict_acgt[j])
    sequences_num.append(list_a)
sequence_array = np.array(sequences_num)
# 初始化计数器
repeated_count = 0
unique_sequences = set()
dup = False
repeats = set()



# 遍历所有碱基序列片段，计算重复率
for i in range(len(sequences)):
    print("i-----------------------------------------", i)
    s =time.time()
    for j in range(i+1, len(sequences)):
        # 检查两个碱基序列片段是否有n个位置相同
        # 将diff置零
        diff = Levenshtein.hamming(sequences[i], sequences[j])
        # 如果比对完了，diff的值小于thresh，说明这两个片段相似,加入到repeats中。
        if diff <= thresh:
            print("here")
            repeats.add(sequences[i])
            repeats.add(sequences[j])
            dup = True
    end = time.time()
    print(s - end)
    # print("j",j)
    if dup == False: #如果遍历完一次。发现dup还是False，说明是独立片段
        unique_sequences.add(sequences[i])
    dup = False

rate = len(repeats) / len(total_read_num)
# 输出结果
print("rate:",rate)
print(f"文件中共有{len(sequences)}个碱基序列片段")
print(f"其中有{len(unique_sequences)}种不重复的碱基序列片段")
print(f"文件中有{repeated_count}个碱基序列片段是重复的")
"""

python 写完成碱基序列片段重复率计算。
1、碱基序列片段是一段字符串例如“AACCATCGAAGCT”长度为100，当然长度也可以是一个变量。
2、碱基序列片段保存在一个txt文件中。需要读入。读入后保存到列表中，一般一个txt文件有30万个片段。
4、设置一个阈值n，当两个序列片段有n个位置不相同时，认为这两个碱基序列片段不是重复的，否则就是重复的。
5、需要计算出文件中有多少个片段是重复的。
6、计算出文件中有多少种片段，重复的片段只算一种。
7、计算重复率
不需要全部比对，有n个不同就可以认为不是重复片段了。

"""
