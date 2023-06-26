import numpy as np
import glob
"""
NOT PURE not pure,
mapping -1 and label 5 are included.
use for test


the final shape is:  readsNUm x  cycle x channel(4),横着的是seq_len，
use msk to pick the point 
make the label npy


"""
list_reads = []
list_channel = []
list_cycle = []
cycle = 100
machinename = '21'
# runname='R004C100'

total_30path = glob.glob(r"E:\code\python_PK\callbase\datasets\21\Res\Lane01\deepLearnData\*.npy")
for name in total_30path:
    runname = name[-17:-9]

    if runname == "R001C001" :
        continue
    print(runname)



    totalpath_A  = glob.glob(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\*\intensity_norm\{}_A.npy".format(machinename,runname))
    msk_path = r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\{}_mask.npy".format(machinename,runname)
    msk = np.load(msk_path).astype(int)
    totalReadsNum = np.sum(abs(msk)) # +1和-1都需要，因此是不pure。
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
        m = 0

        for path in listPath:
            listimg.append(np.load(path).flatten()[np.newaxis,:])
        # listimg has 4 imgs, make it an array
        imgTotal = np.concatenate(listimg)  #  4,h,w
        msk = msk.flatten()
        idx = np.where(msk!=0)
        result_ = imgTotal[:,idx].transpose([2,1,0]) # numreads,1,channel  .squeeze().transpose() # numreads,channel
        result_label = label[idx]

        dataArray[:,n:n+1,:] = result_ # m: reads in one picture,n：n-th cycle,
        label_array[:,n] = result_label
        #label_array[m,n] = label[i,j]
        n = n + 1 #

        print("m is {},totalreads is {},n is {}".format(m,totalReadsNum,n))
    np.save("flatten/val/notpure/{}_{}.npy".format(machinename,runname), dataArray)
    np.save("flatten/val/notpure/{}_{}_label.npy".format(machinename,runname), label_array)
#np.save("21_R001C001_label.npy",label_array)

#246138 R002C034_A 去除label 为5和未mapping的




