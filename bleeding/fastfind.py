import  os
import glob
import numpy as np
zerosarrar = np.zeros((2160,4096))
def lvbo(input_matrix,filter_size = 3):
    out_readId_list = []
    a = 0
    for i in range(filter_size//2, input_matrix.shape[0]-filter_size//2):
        for j in range(filter_size//2, input_matrix.shape[1]-filter_size//2):
            # 获取元素周围的元素
            if  input_matrix[i, j] != 0:
                a = a101
                #print(a)
                elements = input_matrix[i-filter_size//2:i+filter_size//2+1, j-filter_size//2:j+filter_size//2+1].copy().flatten()
                # 对元素值进行排序，取中间值作为输出元素值
                elements = -np.sort(-elements) #降序排列
                second_value = elements[1]
                #如果这个值不为0，则这个点不是孤立点：
                if  second_value == 0:
                    #孤立点：
                    out_readId_list.append([input_matrix[i,j],j,i])#按x,y 坐标的方式进行保存。
                    # print("x,y: ",j,i)
    return  out_readId_list#返回这个矩阵的label,i,j#i属于列，属于行
coordDirTotal = r"E:\code\python_PK\callbase\datasets\highDens_08\result_outcorrd\Lane01\sfile"
for coordFile in glob.glob(os.path.join(coordDirTotal, "R001C001.temp")):#读取坐标。
    if '_' not in os.path.basename(coordFile):
        #print(coordFile)
        FOV = os.path.splitext(os.path.basename(coordFile))[0]
        peak_sub = np.loadtxt(coordFile, skiprows=2)
        peak = np.around(peak_sub).astype(int)
        peakT = peak.T
        for readId,peakTemp in enumerate(peak):
            zerosarrar[peakTemp[1],peakTemp[0]] = readId+1
        out_readId_list = lvbo(zerosarrar)
        list_array = np.array(out_readId_list).astype(int).transpose((1,0))

        coord = peak_sub[list_array[0]-1].transpose((1,0))
        label_coord = np.concatenate([(list_array[0]-1)[np.newaxis,:],coord]).transpose((1,0))
        label_coord = label_coord[np.argsort(label_coord[:,0])]
        print("ddd")

np.savetxt(r"E:\code\python_PK\callbase\datasets\highDens_08\result_outcorrd\Lane01\sfile\nearReadSet\R001C001_conv.npy", np.array(label_coord))
print(len(out_readId_list))




"""
coordDirTotal = r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile"
outPath = r"E:\code\python_PK\callbase\datasets\highDens\result_outcorrd\Lane01\sfile\nearReadSet"
threshDis = 3.5
OulierSuffix = "Outlier_Id.txt"
def getOutlier(coordDirTotal, outPath):
    ### get Outlier for each FOV
    for coordFile in glob.glob(os.path.join(coordDirTotal, "*.temp")):
        if '_' not in os.path.basename(coordFile):
            print(coordFile)
            FOV = os.path.splitext(os.path.basename(coordFile))[0]
            peakT = np.loadtxt(coordFile, skiprows=2)
            peakT = peakT.astype(np.float32)
            peak = peakT.T
            nearReadSet = []  ### reacord Outlier readId
            for readId, peakTemp in enumerate(peakT):
                #### calculate each
                nearPos = np.where((peak[0] < peakTemp[0] + threshDis) & (peak[0] > peakTemp[0] - threshDis) &
                         (peak[1] < peakTemp[1] + threshDis) & (peak[1] > peakTemp[1] - threshDis))[0]
                if len(nearPos) == 1:
                    nearReadSet.append([readId, peakTemp[0], peakTemp[1]])
                    continue
                # else:
                #     difX = np.square(peak[0][nearPos] - peakTemp[0])
                #     difY = np.square(peak[1][nearPos] - peakTemp[1])
                #     disTemp = np.sqrt(difX + difY)  # np.sqrt(difX + difY) #这些点里有一个是自己。
                #
                #     nearNum = len(np.where(disTemp < threshDis)[0])
                #
                #     if nearNum == 1: #当只有自己的时候，则还是把自己加入到列表里去。
                #         #print(np.sort(disTemp))
                #         nearReadSet.append([readId, peakTemp[0], peakTemp[1]])
            #### out nearRead Id to file
            print(len(nearReadSet))
            np.savetxt(os.path.join(outPath, FOV + OulierSuffix), np.array(nearReadSet))
           
getOutlier(coordDirTotal,outPath)



"""