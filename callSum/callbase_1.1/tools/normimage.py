import numpy as np
import glob

# E:\code\python_PK\callbase\datasets\21\Res\Lane01\deepLearnData\Cyc001\intensity

# 将所有的亮度矩阵都归一化，并另外保存，在intensity_norm里
# 将归一化的图象里和label里为5的区域。进行掩盖。
def dellist(list_i,value):
    list_out = []
    for i in list_i:
        if i != 0:
            list_out.append(i)
    return list_out
def normarray(a,p1,p99):
    outa = (a-p1)/(p99-p1)
    return outa
paths = glob.glob(r"E:\code\python_PK\callbase\datasets\21\Res\Lane01\deepLearnData\*\intensity\*.npy")
mask = np.load(r"E:\code\python_PK\callbase\datasets\21\Res\Lane01\deepLearnData\R001C001_mask.npy")
for path in paths:
    print(path)
    array_  = np.load(path)
    a,b = array_.shape
    array_2 = array_.reshape(1,a*b)[0,:]
    array_list = array_2.tolist()
    array_list2 = dellist(array_list,0)
    array_list_sorted = sorted(array_list2) #升序排列
    list_len = len(array_list_sorted)
    p1_idx = int(0.01*list_len)
    p99_idx = int(0.99*list_len)
    p1 = array_list_sorted[p1_idx]
    p99 = array_list_sorted[p99_idx]
    print("p1:", p1)
    print("p99", p99)
    outarray = normarray(array_,p1,p99)
    print("here")



 #采用mask读取：
    # array_illumin = np.multiply(array_,abs(mask))
    # array_illumin2 =  array_illumin.reshape(1,a*b)[0,:]
    # array_illumin2_list = array_illumin2.tolist()
    # array_illumin2_list2 = dellist(array_illumin2_list, 0)
    # array_illumin_list_sorted = sorted(array_illumin2_list2)
    # list_illumin_len = len(array_illumin_list_sorted)
    # print("list:{},list_ill:{}".format(list_len,list_illumin_len))
    # p1_ill_idx = int(0.01*list_illumin_len)
    # p99_ill_idx = int(0.99*list_illumin_len)
    # p1_ill = array_illumin_list_sorted[p1_ill_idx]
    # p99_ill = array_illumin_list_sorted[p99_ill_idx]
    # print("p1_ill:", p1_ill)
    # print("p99_ill:", p99_ill)


