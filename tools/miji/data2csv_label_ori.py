import numpy as np
import glob
"""
the final shape is:  readsNUm x  cycle x channel(4),横着的是seq_len，
use msk to pick the point 
make the label npy

"""
list_reads = []
list_channel = []
list_cycle = []
cycle = 100
totalpath_A  = glob.glob(r"E:\code\python_PK\callbase\datasets\21\Res\Lane01\deepLearnData\*\intensity_norm\R001C001_A.npy")
msk_path = r"E:\code\python_PK\callbase\datasets\21\Res\Lane01\deepLearnData\R001C001_mask.npy"
msk = np.load(msk_path).astype(int)
totalReadsNum = np.sum(abs(msk)) # as the first dimension of the array
dataArray = np.zeros((totalReadsNum,100,4))
label_array = np.zeros((totalReadsNum,100)) #label 只是一个通道 0-5。可以说是2维的。
n = 0
num_5 = 0
for pathA in totalpath_A:
    pathC = pathA.replace("A","C")
    pathG = pathA.replace("A","G")
    pahtT = pathA.replace("A","T")
    listPath = [pathA,pathC,pathG,pahtT]
    pathLabel = pathA.replace("intensity_norm","label").replace("A","label")
    label = np.load(pathLabel).astype(int)
    listimg = []
    m = 0

    for path in listPath:
        listimg.append(np.load(path)[np.newaxis,:])
    # listimg has 4 imgs, make it an array
    imgTotal = np.concatenate(listimg)  #  4,h,w
    for i in range(msk.shape[0]):
        for j in range(msk.shape[1]):
            # if label[i,j]== 5:
            #     num_5 = num_5 + 1
            #     continue

            if msk[i][j] == 1 and label[i,j]== 5  :
                dataArray[m,n,:] = imgTotal[:,i,j] # m: reads in one picture,n：n-th cycle,
                label_array[m,n] = label[i,j]
                m = m + 1
    print("label 5 num: ", num_5)
    n = n + 1 #
    num_5 = 0

    print("m is {},totalreads is {},n is {}".format(m,totalReadsNum,n))
np.save("test_21_R001C001.npy",dataArray)
np.save("test_21_R001C001_label.npy",label_array)

#246138 R002C034_A 去除label 为5和未mapping的




