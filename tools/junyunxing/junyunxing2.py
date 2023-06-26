import  os
import glob
import numpy as np
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance

zerosarrar = np.zeros((2200,4200))

# kernel = np.array([[1, 1, 1],
#                    [1, 1, 1],
#                    [1, 1, 1]])
kernel = np.ones((23,23))

coordDirTotal = r"C:\Users\Administrator\Documents\WeChat Files\wxid_izeosircs0gj22\FileStorage\File\2023-05"
filename = "R001C001_A##R001C001_C.txt"
for coordFile in glob.glob(os.path.join(coordDirTotal,filename)):#读取坐标。

    # if '_' not in os.path.basename(coordFile):
        #print(coordFile)
        FOV = os.path.splitext(os.path.basename(coordFile))[0]
        #peak_sub = np.loadtxt(coordFile, skiprows=2)
        peak_sub = np.loadtxt(coordFile)
        peak = np.around(peak_sub).astype(int)
        peakT = peak.T
        for readId,peakTemp in enumerate(peak):
            zerosarrar[peakTemp[1],peakTemp[0]] = 1

        # out_readId_list = conv(zerosarrar,peak_sub)
        result = convolve2d(zerosarrar, kernel, mode='same', boundary='fill', fillvalue=0)
        arr = result.flatten()
        plt.hist(arr,bins = np.arange(1,100))

        # 显示图像

        plt.savefig(rf"hist_{filename.split('.')[0]}.png")
        plt.show()
        img = cv2.applyColorMap(result.astype(np.uint8), cv2.COLORMAP_JET)

        # # 显示图像
        # cv2.imshow('Pseudocolor Image', img)
        # cv2.waitKey(0)
        #
        # # 保存图像
        # cv2.imwrite(rf'pseudocolor_{filename.split(".")[0]}.png', img)