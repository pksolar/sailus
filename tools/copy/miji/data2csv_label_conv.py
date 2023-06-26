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
totalpath_A  = glob.glob(r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\*\intensity_norm\R002C034_A.npy")
msk_path = r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\R002C034_mask.npy"
msk = np.load(msk_path).astype(int)
totalReadsNum = np.sum(abs(msk)) # as the first dimension of the array
dataArray = np.zeros((totalReadsNum,100,4))
label_array = np.zeros((totalReadsNum,100)) #label 只是一个通道 0-5。可以说是2维的。
n = 0
for pathA in totalpath_A:
    pathC = pathA.replace("A","C")
    pathG = pathA.replace("A","G")
    pahtT = pathA.replace("A","T")
    listPath = [pathA,pathC,pathG,pahtT]
    pathLabel = pathA.replace("intensity_norm","label").replace("A","label")
    label = np.load(pathLabel)
    listimg = []
    m = 0

    for path in listPath:
        listimg.append(np.load(path)[np.newaxis,:])
    # listimg has 4 imgs, make it an array
    imgTotal = np.concatenate(listimg)  #  4,h,w
    #先把图像分成nxn的块，再对块进行统计：
    n_size = 7
    height,width = label.shape
    h_num = np.floor(height/n_size).astype(int)
    w_num = np.floor(width/n_size).astype(int)
    for ii in range(h_num):
        for jj in range(w_num):
            rowmin = n_size*ii
            rowmax = n_size*(ii+1)
            colmin = n_size * jj
            colmax = n_size * (jj + 1)
            if rowmax > height:
                rowmax = height
            if colmax > width:
                colmax = width

            roi_pic = imgTotal[:,rowmin:rowmax,colmin:colmax]
            roi_msk = msk[rowmin:rowmax,colmin:colmax]
            roi_label = label[rowmin:rowmax,colmin:colmax]


            for i in range(roi_msk.shape[0]):
                for j in range(roi_msk.shape[1]):
                    if roi_msk[i][j] !=0:#== 1 and roi_label[i,j]!= 5  :

                        dataArray[m,n,:] = roi_pic[:,i,j] # m: reads in one picture,n：n-th cycle,
                        label_array[m,n] = roi_label[i,j]
                        m = m + 1
    n = n + 1 #

    print("m is {},totalreads is {},n is {}".format(m,totalReadsNum,n))
np.save("08_R002C034_conv_notpure.npy",dataArray)
np.save("08_R002C034_conv_notpure_label.npy",label_array)

#246138 R002C034_A 去除label 为5和未mapping的

#224496 是R001C001去除msk -1 和label 5




