import numpy as np
with open(r"E:\data\resize_test\08_resize_ori\res\Lane01\Lane01_fastq.fq",'r') as f:
    content = f.readlines()
    # idx =  range(1:328478 * 4,4)
    result = content[1::4]
with open("08_fastq.txt",'w') as f:
    for item in result:
        f.write(item)


