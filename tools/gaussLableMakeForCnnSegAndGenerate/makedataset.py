import os
import glob
import numpy as np
import cv2
from scipy.ndimage import grey_dilation, generate_binary_structure
"""

"""
def gaussReplace(matrix,kernel):
    """
    matrix：label矩阵 0,1,2,3,4,5只需要1,2,3,4

    """
    #matrix = np.load("label.npy")
    # 将数字为1的位置标记为1，其余位置标记为0
    ones_matrix = np.where(matrix == 1, 1, 0).astype(np.float64)

    # 将数字为2的位置标记为1，其余位置标记为0
    twos_matrix = np.where(matrix == 2, 1, 0)

    # 将数字为3的位置标记为1，其余位置标记为0
    threes_matrix = np.where(matrix == 3, 1, 0)

    # 将数字为4的位置标记为1，其余位置标记为0
    fours_matrix = np.where(matrix == 4, 1, 0)

    matrix_total = np.stack([ones_matrix,twos_matrix,threes_matrix,fours_matrix])
    for i in range(4):
        temp_matrix = matrix_total[i,:,:]
        indices = np.argwhere(temp_matrix == 1)
        # 将小矩阵中的值与大矩阵对应位置的值进行比较，并更新大矩阵的值
        for index in indices:
            i, j = index
            start_i, start_j = max(i - 2, 0), max(j - 2, 0)
            end_i, end_j = min(i + 3, temp_matrix.shape[0]), min(j + 3, temp_matrix.shape[1])
            patch = kernel[start_i - i + 2:end_i - i + 2, start_j - j + 2:end_j - j + 2]
            big_matrix_patch = temp_matrix[start_i:end_i, start_j:end_j]
            updated_values = np.where(patch > big_matrix_patch, patch, big_matrix_patch)
            temp_matrix[start_i:end_i, start_j:end_j] = updated_values
            # print(updated_values)
            # print(temp_matrix[start_i:end_i, start_j:end_j])

    return matrix_total

sigma = 1.2
size = 5
center = size // 2
x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
kernel /= kernel.sum()
rate = 1/kernel[2,2]
kernel = rate*kernel

label_dir = "featurelabel" #img2base  val
msk_dir = "featuremsk"
machine_name = '08'
fov = 'R001C001'
#E:\code\python_PK\callbase\datasets\30\Image\result\Image\Lane01\Cyc001
# E:\data\resize_test\08_resize_ori\res\Lane01\deepLearnData\*\label
path_labels = glob.glob(fr"E:\data\resize_test\08_resize_ori\res\Lane01\deepLearnData\*\label\{fov}_label.npy")
# path_label_alone = r"E:\code\python_PK\callbase\datasets\30\Res\Lane01\deepLearnData1\Cyc001\label\R001C001_label.npy"
mask_path =rf"E:\data\resize_test\08_resize_ori\res\Lane01\deepLearnData\{fov}_mask.npy"

msk = np.load(mask_path)
msk[msk==1] = 0
msk[msk<0] = 1
struct = generate_binary_structure(2, 2)
msk = grey_dilation(msk, footprint=struct)
msk = grey_dilation(msk,footprint=struct)
msk[msk==0] = 2
msk[msk==1] = 0
msk[msk ==2 ]=1
os.makedirs(f"{msk_dir}", exist_ok=True)
np.save(f"{msk_dir}/{machine_name}_{fov}.npy", msk)


for path_label in path_labels :
    #namelist = ['_C', '_G', '_T']
    print("oriname:", path_label)
    #for name in namelist:
    idx = 1
    label = np.load(path_label)
    label_new = gaussReplace(label,kernel)
    cycname = path_label.split("\\")[-3]
    # print(imgname)

    os.makedirs(f"{label_dir}", exist_ok=True)

    # os.makedirs("img2base/lab", exist_ok=True)

    np.save(f"{label_dir}/{machine_name}_{fov}_{cycname}.npy", label_new)

    idx = idx + 1
    print("idx：", idx)



