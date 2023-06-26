import numpy as np
import glob
"""
Pure 
mapping 1 used
use for training

the final shape is:  readsNUm x  cycle x channel(4),横着的是seq_len，
use msk to pick the point 
make the label npy
"""

list_reads = []
list_channel = []
list_cycle = []
cycle = 100
machinename = '30'
runname='R001C001'

totalpath_A  = glob.glob(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\{}_A.npy".format(machinename,runname))
msk_path = r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\{}_mask.npy".format(machinename,runname)
msk = np.load(msk_path).astype(int)
msk[msk==-1]=0 #
totalReadsNum = np.sum(abs(msk)) # only msk ==1:
dataArray = np.zeros((totalReadsNum,100,4))
label_array = np.zeros((totalReadsNum,100)) #label 只是一个通道 0-5。可以说是2维的。
n = 0
for pathA in totalpath_A:
    pathC = pathA.replace("A","C")
    pathG = pathA.replace("A","G")
    pahtT = pathA.replace("A","T")
    listPath = [pathA,pathC,pathG,pahtT]
    pathLabel = pathA.replace("intensity_norm","label").replace("A","label")
    label = np.load(pathLabel).flatten()
    listimg = []

    for path in listPath:
        listimg.append(np.load(path).flatten()[np.newaxis,:])
    # listimg has 4 imgs, make it an array
    imgTotal = np.concatenate(listimg)  #  4,h,w
    msk = msk.flatten() #
    idx = np.where(msk!=0) #不为0的地方，
    result_ = imgTotal[:,idx].transpose([2,1,0]) # numreads,1,channel  .squeeze().transpose() # numreads,channel
    result_label = label[idx]

    dataArray[:,n:n+1,:] = result_ # m: reads in one picture,n：n-th cycle,
    label_array[:,n] = result_label
    #label_array[m,n] = label[i,j]
    n = n + 1 #

    print("totalreads is {},n is {}".format(totalReadsNum,n))
np.save("flatten/pure/{}_{}_pure.npy".format(machinename,runname), dataArray)
np.save("flatten/pure/{}_{}_label_pure.npy".format(machinename,runname), label_array)
#np.save("21_R001C001_label.npy",label_array)
#246138 R002C034_A 去除label 为5和未mapping的




