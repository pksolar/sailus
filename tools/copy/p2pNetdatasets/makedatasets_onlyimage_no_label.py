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
paths = glob.glob(r"E:\code\python_PK\callbase\datasets\highDens\Cyc003\*_A.tif")

patch_size = 512
hn = np.floor(2160/512).astype(int)
wn = np.floor(4096/512).astype(int)
idx = 1737
for path in paths :
    namelist = ['_A', '_C', '_G', '_T']
    print("oriname:", path)

    for name in namelist:

        name_path = path.replace("_A", name)

        print(name_path)

        img = cv2.imread(name_path,0)

        print("img：",img.shape)

        for i in range(hn-1):
            for j in range(wn-1):
                rowmin = i * patch_size
                rowmax = (i + 1)*patch_size
                colmin = j * patch_size
                colmax = (j+1) * patch_size
                imgcrop = img[rowmin:rowmax,colmin:colmax].copy()

                imgname = name_path.split("\\")[-1].replace(".tif","")
                #print(imgname)

                cv2.imwrite("highdense/images/img{:04d}.jpg".format(idx), imgcrop)

                idx = idx + 1
                print("idx：", idx)








#
# img_to_draw = cv2.cvtColor(np.array(imgcrop), cv2.COLOR_RGB2BGR)
#             if "A" in imgname:
#                for ii in range(512):
#                    for jj in range(512):
#                        if labelcrop[ii,jj] == 1:
#                             img_to_draw = cv2.circle(img_to_draw, (ii,jj), 1, (0, 0, 255), -1)
