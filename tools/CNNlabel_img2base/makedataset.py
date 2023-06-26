import os
import glob
import numpy as np
import cv2
from scipy.ndimage import grey_dilation, generate_binary_structure
"""

"""

def norm(arr):  # 对每个通道各自做归一化。应该是做99归一化
    for i in range(arr.shape[0]):
        channel = arr[i]
        min_val = np.amin(channel)
        max_val = np.amax(channel)
        arr[i] = (channel - min_val) / (max_val - min_val)
    return arr

def gaussReplace(matrix,base_type):
    ones_matrix = np.where(matrix == base_type, 1, 0).astype(np.float64)
    struct = generate_binary_structure(2, 2)
    ones_matrix = grey_dilation(ones_matrix, footprint=struct)
    ones_matrix = grey_dilation(ones_matrix, footprint=struct)
    return ones_matrix

kernel = np.ones((5,5))
val = "img2base" #img2base  val
machine_name = '30'
fov = 'R001C001'
#E:\code\python_PK\callbase\datasets\30\Image\result\Image\Lane01\Cyc001
paths = glob.glob(fr"E:\code\python_PK\callbase\datasets\{machine_name}\Res\Image\Lane01\*\{fov}_A.jpg")
path_labels = glob.glob(fr"E:\code\python_PK\callbase\datasets\{machine_name}\Res\Lane01\deepLearnData\*\label\{fov}_label.npy")
# path_label_alone = r"E:\code\python_PK\callbase\datasets\30\Res\Lane01\deepLearnData1\Cyc001\label\R001C001_label.npy"
mask_path =rf"E:\code\python_PK\callbase\datasets\{machine_name}\Res\Lane01\deepLearnData\{fov}_mask.npy"

msk = np.load(mask_path)
msk[msk<0] = 0
struct = generate_binary_structure(2, 2)
msk = grey_dilation(msk, footprint=struct)
msk = grey_dilation(msk,footprint=struct)

# patch_size1 = 2160
# patch_size2 = 4096
# hn = np.floor(2160/patch_size1).astype(int)
# wn = np.floor(4096/patch_size2).astype(int)

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
    label_npy[0,:,:] = gaussReplace(label,1)
    label_npy[1,:,:] = gaussReplace(label, 2)
    label_npy[2,:,:] = gaussReplace(label, 3)
    label_npy[3,:,:] = gaussReplace(label, 4)

    cycname = path.split("\\")[-2]



    # print(imgname)

    os.makedirs(f"{val}/{machine_name}_{fov}_imgdata", exist_ok=True)
    os.makedirs(f"{val}/{machine_name}_{fov}_label", exist_ok=True)
    os.makedirs(f"{val}/{machine_name}_{fov}_mask", exist_ok=True)
    # os.makedirs("img2base/lab", exist_ok=True)
    np.save(f"{val}/{machine_name}_{fov}_imgdata/{cycname}.npy", norm(img_npy))
    np.save(f"{val}/{machine_name}_{fov}_label/{cycname}.npy", label_npy)
    np.save(f"{val}/{machine_name}_{fov}_mask/{fov}.npy", msk)
    idx = idx + 1
    print("idx：", idx)



