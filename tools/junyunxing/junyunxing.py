import  os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
width = 4200
height = 2700
zerosarrar = np.zeros((height,width))

def k_min(matrix,peak_sub,filename,size = 11):
    # 创建一个全0矩阵，用于存储每个点到最近的非0点的距离
    size = int((size-1)/2)
    distances = np.zeros_like(matrix, dtype=float)

    # 使用掩码操作获取非0点的坐标
    rows, cols = np.nonzero(matrix)
    result_list = []
    # 对每个非0点进行处理
    for i, j in zip(rows, cols):
        if i > size and j > size and i < height-size and j < width-size:
            #得到这个位置的id,以及该id下的亚像素坐标：
            centerId = int(matrix[i,j])
            center_lco = peak_sub[centerId,:]  #x,y形式。j,i 形式

            # 创建一个7x7的子矩阵，用于计算距离
            sub_matrix = matrix[max(i - size, 0):i + size+1, max(j - size, 0):j + size+1]

            # 创建一个掩码，将子矩阵中的0排除在计算之外
            mask = sub_matrix != 0
            mask[size,size] = False
            # 如果子矩阵中只有0，将距离设为一个大数
            t = np.sum(mask)
            if np.sum(np.sum(mask)) == 0:
                distances[i, j] = np.inf
            else:
                list_distance = []
                for ii in range(np.sum(mask)):
                      near_y_idx =  np.where(mask)[0][ii] - size + i
                      near_x_idx =  np.where(mask)[1][ii] - size + j
                      near_id = int(matrix[near_y_idx,near_x_idx])
                      near_loc = peak_sub[near_id,:]
                      distance =((center_lco[0]-near_loc[0])**2+(center_lco[1]-near_loc[1])**2)**0.5
                      list_distance.append(distance)
                min_distance = np.min(np.array(list_distance))
                result_list.append(min_distance)
    # 将距离中的inf替换为0，方便后续的直方图绘制
    # distances[distances == np.inf] = 0
    # np.save("distance.npy",distances)
    # # 绘制直方图
    # arr = distances[np.nonzero(distances)].ravel()
    # print(np.var(arr))
    # 按照区间宽度 0.1 统计直方图
    plt.hist(result_list, bins=100)
    plt.savefig(rf"hist_{filename.split('.')[0]}.png")
    plt.show()
    # hist, bins = np.histogram(np.array(result_list), bins=np.arange(0, np.max(np.array(result_list)) + 0.01, 0.01))
    #
    # # 绘制直方图
    # plt.bar(bins[:-1], hist, width=1)
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.show()
# coordDirTotal = r"E:\data\resize_test\30_resize_ori\res\Lane01\sfile\\"
coordDirTotal = r"C:\Users\Administrator\Documents\WeChat Files\wxid_izeosircs0gj22\FileStorage\File\2023-05\\"
filename = "R002C106_A##R002C106_G.txt"
for coordFile in glob.glob(os.path.join(coordDirTotal, filename)):#读取坐标。

    if '_'  in os.path.basename(coordFile):
        #print(coordFile)
        FOV = os.path.splitext(os.path.basename(coordFile))[0]
        # peak_sub = np.loadtxt(coordFile, skiprows=2)
        peak_sub = np.loadtxt(coordFile)
        peak = np.around(peak_sub).astype(int)
        peakT = peak.T
        for readId,peakTemp in enumerate(peak):
            zerosarrar[peakTemp[1],peakTemp[0]] = readId

        out_readId_list = k_min(zerosarrar,peak_sub,filename)

        print("ddd")