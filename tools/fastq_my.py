import numpy as np
import glob
import cv2
import fastQ
import time
img_path = glob.glob(r"E:\code\python_PK\image2base\FIDTM-master-basecall\result\Lane01\*\R001C001_A.tif")


dictacgt = {1: "A", 2: "C", 3: "G", 4: "T"}
str_acgt = ""


paths =glob.glob(r"E:\code\python_PK\image2base\FIDTM-master-basecall\result\Lane01\*\R001C001_A.tif")
# path_label_alone = r"E:\code\python_PK\callbase\datasets\30\Res\Lane01\deepLearnData1\Cyc001\label\R001C001_label.npy"
mask_path =r"E:\code\python_PK\callbase\datasets\30\Res\Lane01\deepLearnData1\R001C001_mask.npy"
msk = np.load(mask_path)
msk = abs(msk)
predict_list  = []
s = time.time()
for path in paths:
    #namelist = ['_C', '_G', '_T']

    #for name in namelist:
    idx = 1
    img_npy = np.zeros([4,2160,4096])

    name_path_C = path.replace("_A", '_C')
    name_path_G = path.replace("_A", '_G')
    name_path_T = path.replace("_A", '_T')


    imgA = cv2.imread(path,0)
    imgC = cv2.imread(name_path_C, 0)
    imgG = cv2.imread(name_path_G, 0)
    imgT = cv2.imread(name_path_T, 0)

    img_npy[0,:,:] = imgA
    img_npy[1, :, :] = imgC
    img_npy[2, :, :] = imgG
    img_npy[3, :, :] = imgT


    #在通道内取最值。再用mask覆盖一下。
    call = np.argmax(img_npy,axis=0) + 1
    # 与mask相乘。
    result_npy = call * msk # 1 2 3 4
    print("hello world")


    cycname = path.split("\\")[-2]


    k = 0
    if cycname == "Cyc001":
        for i in range(result_npy.shape[0]):
            for j in range(result_npy.shape[1]):
                if result_npy[i,j] != 0 :
                        str_acgt = dictacgt[result_npy[i ,j]]
                        predict_list.append(str_acgt)
                        # print(len(predict_list))
                        # print(predict_list[:3])
    else:
        for i in range(result_npy.shape[0]):
            for j in range(result_npy.shape[1]):
                if result_npy[i, j] != 0:
                    predict_list[k] = predict_list[k] + dictacgt[result_npy[i ,j]]
                    k += 1
                    # print(len(predict_list))
                    # print(predict_list[:3])
    # if cycname == "Cyc003":
    #     break
    print(cycname)

timestr = time.strftime("%Y%m%d-%H%M%S")
fastQ.writeFq('fastq/fast30_{}.fq'.format(timestr), predict_list, 'ROO1C001')
end = time.time()
print("time:" ,end-s)
