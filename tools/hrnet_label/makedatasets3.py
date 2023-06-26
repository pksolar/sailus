import cv2
"""
找到一个数据集即可：
把一张图切成n个512x512，同时label矩阵也被切成512x512
输出label中非0的位置信息，label中：0,1,2,3,4,5。

"""
import  numpy
import glob
import numpy as np

#E:\code\python_PK\callbase\datasets\30\Image\result\Image\Lane01\Cyc001
paths = glob.glob(r"E:\code\python_PK\callbase\datasets\30\Image\result\Image\Lane01\Cyc001\*_A.jpg")
path_labels = glob.glob(r"E:\code\python_PK\callbase\datasets\30\Res\Lane01\deepLearnData1\Cyc001\label\*label.npy")
patch_size = 512
hn = np.floor(2160/512).astype(int)
wn = np.floor(4096/512).astype(int)
idx = 1
for (path,path_label) in zip(paths,path_labels) :
    namelist = ['A','C','G','T']
    print("oriname:",path)
    for name in namelist:
        name_path = path.replace("_A","_"+name)

        print(name_path)
        print(path_label)
    # img = cv2.imread(path,0)
    # label = np.load(path_label)
    # print("img：",img.shape)
    # print("label: ",label.shape)
    #
    # imgname = path.split("\\")[-1]
    # print(imgname)
    # poslist = []
    # img_to_draw = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # if "A" in imgname:
    #     for ii in range(2160):
    #         for jj in range(4096):
    #             if label[ii, jj] == 1:
    #                 #img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    #                 img_to_draw = cv2.circle(img_to_draw, (jj, ii), 4, (0, 0, 255), -1)
    #                 cv2.imshow("i", img_to_draw)
    #                 cv2.imshow("ori", img)
    #                 cv2.imwrite("point.jpg",img_to_draw)
    #                 cv2.imwrite("ori.jpg",img)
    #                 idx = idx + 1
    #                 print("idx：", idx)
    #                 cv2.waitKey(0)
    #
    #







#
# img_to_draw = cv2.cvtColor(np.array(imgcrop), cv2.COLOR_RGB2BGR)
#             if "A" in imgname:
#                for ii in range(512):
#                    for jj in range(512):
#                        if labelcrop[ii,jj] == 1:
#                             img_to_draw = cv2.circle(img_to_draw, (ii,jj), 1, (0, 0, 255), -1)
