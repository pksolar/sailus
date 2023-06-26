import os
import glob
import numpy as np
import cv2
from scipy.ndimage import grey_dilation, generate_binary_structure
"""

"""
def gaussReplace(matrix,base_type,kernel):
    ones_matrix = np.where(matrix == base_type, 1, 0).astype(np.float64)
    indices = np.argwhere(ones_matrix == 1)
    # 将小矩阵中的值与大矩阵对应位置的值进行比较，并更新大矩阵的值
    for index in indices:
        i, j = index
        start_i, start_j = max(i - 2, 0), max(j - 2, 0)
        end_i, end_j = min(i + 3, ones_matrix.shape[0]), min(j + 3, ones_matrix.shape[1])
        patch = kernel[start_i - i + 2:end_i - i + 2, start_j - j + 2:end_j - j + 2]
        big_matrix_patch = ones_matrix[start_i:end_i, start_j:end_j]
        updated_values = np.where(patch > big_matrix_patch, patch, big_matrix_patch)
        ones_matrix[start_i:end_i, start_j:end_j] = updated_values
    return ones_matrix

sigma = 1.2
size = 5
center = size // 2
x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
kernel /= kernel.sum()
rate = 1/kernel[2,2]
kernel = rate*kernel
val = "img2base" #img2base  val

#E:\code\python_PK\callbase\datasets\30\Image\result\Image\Lane01\Cyc001
paths = glob.glob(r"E:\code\python_PK\callbase\datasets\30\Res\Image\Lane01\*\R001C001_A.jpg")
path_labels = glob.glob(r"E:\code\python_PK\callbase\datasets\30\Res\Lane01\deepLearnData1\*\label\R001C001_label.npy")
# path_label_alone = r"E:\code\python_PK\callbase\datasets\30\Res\Lane01\deepLearnData1\Cyc001\label\R001C001_label.npy"
mask_path =r"E:\code\python_PK\callbase\datasets\30\Res\Lane01\deepLearnData1\R001C001_mask.npy"
msk = np.load(mask_path)
msk[msk<0] = 0
msk_alone = msk.copy()

struct = generate_binary_structure(2, 2)
msk = grey_dilation(msk, footprint=struct)
msk = grey_dilation(msk,footprint=struct)

patch_size = 256
hn = np.floor(2160/patch_size).astype(int)
wn = np.floor(4096/patch_size).astype(int)

for (path,path_label) in zip(paths,path_labels) :
    #namelist = ['_C', '_G', '_T']
    print("oriname:", path_label)
    #for name in namelist:
    idx = 1
    img_npy = np.zeros([4,2160,4096])
    label_npy = np.zeros([4,2160,4096])
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

    label = np.load(path_label)
    label_npy[0,:,:] = gaussReplace(label,1,kernel)
    label_npy[1,:,:] = gaussReplace(label, 2,kernel)
    label_npy[2,:,:] = gaussReplace(label, 3,kernel)
    label_npy[3,:,:] = gaussReplace(label, 4,kernel)

    cycname = path.split("\\")[-2]
    if cycname == "Cyc020":
        break
    for i in range(hn-1):
        for j in range(wn-1):
            rowmin = i * patch_size
            rowmax = (i + 1)*patch_size
            colmin = j * patch_size
            colmax = (j+1) * patch_size
            imgcrop = img_npy[:,rowmin:rowmax,colmin:colmax].copy()
            labelcrop = label_npy[:,rowmin:rowmax,colmin:colmax].copy()
            maskcrop = msk[rowmin:rowmax,colmin:colmax].copy()
            msk_alone_crop = msk_alone[rowmin:rowmax,colmin:colmax].copy()

            # print(imgname)


            os.makedirs(f"{val}/imgdata",exist_ok=True)
            os.makedirs(f"{val}/label", exist_ok=True)
            os.makedirs(f"{val}/mask", exist_ok=True)
            os.makedirs(f"{val}/mask2", exist_ok=True)
            # os.makedirs("img2base/lab", exist_ok=True)
            np.save(f"{val}/imgdata/{cycname}_{idx:04d}.npy", imgcrop)
            np.save(f"{val}/label/{cycname}_{idx:04d}.npy", labelcrop)
            np.save(f"{val}/mask/{idx:04d}.npy", maskcrop)
            np.save(f"{val}/mask2/{idx:04d}.npy", msk_alone_crop)
            idx = idx + 1
            print("idx：", idx)



