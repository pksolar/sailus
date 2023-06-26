import numpy as np
import glob
"""
最后的数据的维度是：（101,4,w,h） 或者（101,w,h,4)
w，h由于不同的数据是不同的。



"""
list_reads = []
list_channel = []
list_cycle = []
cycle = 101
machinename = 'highDens'
runname='R001C001'

totalpath_A  = glob.glob(r"E:\code\python_PK\callbase\datasets\{}\Result\Lane01\deepLearnData\*\intensity\{}_A.npy".format(machinename,runname))
msk_path = r"E:\code\python_PK\callbase\datasets\{}\Result\Lane01\deepLearnData\{}_mask.npy".format(machinename,runname)
msk = np.load(msk_path).astype(int)
msk[msk==-1]=0 #不要没有mapping上的数据
totalReadsNum = np.sum(abs(msk)) # only msk ==1: 一个图里有多少个。
bian =int( np.ceil(totalReadsNum**0.5))
numreads  = bian**2
#将totalReadNum宽高比例分成：2:1
dataArray = np.zeros((101,4,numreads))
label_array = np.zeros((101,numreads)) #label 只是一个通道 0-5。可以说是2维的。
n = 0
for pathA in totalpath_A:
    pathC = pathA.replace("A","C")
    pathG = pathA.replace("A","G")
    pahtT = pathA.replace("A","T")
    listPath = [pathA,pathC,pathG,pahtT]
    pathLabel = pathA.replace("intensity","label").replace("A","label")
    label = np.load(pathLabel).flatten()
    listimg = []

    for path in listPath:
        listimg.append(np.load(path).flatten()[np.newaxis,:])
    # listimg has 4 imgs, make it an array
    imgTotal = np.concatenate(listimg)  #  4,hxw
    msk = msk.flatten()
    idx = np.where(msk!=0)
    result_ = imgTotal[:,idx].transpose([1,0,2]) # numreads,1,channel  .squeeze().transpose() # numreads,channel，size
    result_label = label[idx].transpose()

    dataArray[n:n+1,:,:totalReadsNum] = result_ # m: reads in one picture,n：n-th cycle,
    label_array[n,:totalReadsNum] = result_label
    #label_array[m,n] = label[i,j]
    n = n + 1 #
    print("totalreads is {},n is {}".format(totalReadsNum,n))
new_dataArray = dataArray.reshape(101,4,bian,bian)
new_label_array = label_array.reshape(101,bian,bian)

np.save("pailie/{}_{}.npy".format(machinename,runname), dataArray)
np.save("pailie/{}_{}_label.npy".format(machinename,runname), label_array)
    #np.save("21_R001C001_label.npy",label_array)

    #246138 R002C034_A 去除label 为5和未mapping的




