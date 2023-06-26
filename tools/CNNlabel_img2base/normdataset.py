import  numpy as np
import glob
import os

def norm(arr):  # 对每个通道各自做归一化。
    for i in range(arr.shape[0]):
        channel = arr[i]
        min_val = np.amin(channel)
        max_val = np.amax(channel)
        arr[i] = (channel - min_val) / (max_val - min_val)
    return arr
img_paths = glob.glob(r"E:\code\python_PK\tools\CNNlabel_img2base\img2base\21_R001C022_imgdata_seg\*.npy")
rootpath = r"E:\code\python_PK\tools\CNNlabel_img2base\img2base\21_R001C022_imgdataNorm_seg\\"
for path in img_paths:
    arr = np.load(path)
    arr = norm(arr)
    name = os.path.basename(path)
    np.save(rootpath+name,arr)
    print(rootpath+name)

