import numpy as np
import cv2
import glob
import os
"""
读图，4通道,做99归一化，再拼接，3cycle拼接，剪裁成512x512的大小，允许有少量重叠。一个大矩阵可以剪裁成30-40张小矩阵，矩阵格式：12x512x512,。
命名直接采用：数据集代号_crop数.npy

imgPathC = imgPathC.replace("R001C001_A","R001C001_C")
        imgPathG = imgPathC.replace("R001C001_A", "R001C001_G")
        imgPathT = imgPathC.replace("R001C001_A", "R001C001_T")

        #读图：
        imgA = cv2.imread(imgPathA,0)
        imgC = cv2.imread(imgPathC, 0)
        imgG = cv2.imread(imgPathG, 0)
        imgT = cv2.imread(imgPathT, 0)
"""
def read4Img(path):
    name_path_C = path.replace("_A", '_C')
    name_path_G = path.replace("_A", '_G')
    name_path_T = path.replace("_A", '_T')

    imgA = cv2.imread(path, 0)[np.newaxis,:]
    imgC = cv2.imread(name_path_C, 0)[np.newaxis,:]
    imgG = cv2.imread(name_path_G, 0)[np.newaxis,:]
    imgT = cv2.imread(name_path_T, 0)[np.newaxis,:]
    img = np.concatenate([imgA,imgC,imgG,imgT]) #整数chu'fa
    img = norm(img)
    return img

def norm(arr):# 对每个通道各自做归一化。
    for i in range(arr.shape[0]):
        channel = arr[i]
        # print("norm again")
        min_val = np.percentile(channel,1)
        max_val = np.percentile(channel,99)
        arr[i] = (channel - min_val) / (max_val - min_val)
    return arr
saveRootPath = r"E:\data\deepData\train\\"
os.makedirs(saveRootPath+"img",exist_ok=True)
os.makedirs(saveRootPath+"label",exist_ok=True)
os.makedirs(saveRootPath+"msk",exist_ok=True)
rootPath = r"E:\data\resize_test\\"
dirs = os.listdir(rootPath)
h = 2700
w = 5120
for dir_name in dirs:
    print(dir_name)
    machine_name = dir_name.split("_")[0]
    # 读mask,裁剪mask
    msk_path = os.path.join(rootPath, dir_name, "res_deep_intent\Lane01\deepLearnData", "R001C001_mask.npy")
    msk = np.load(msk_path)
    msk_raw = msk.copy()
    patch_size = 256

    rows = int(h / patch_size)
    cols = int(w / patch_size)
    idx = 0
    for k in range(rows + 1):
        for l in range(cols):
            idx += 1
            msk_raw_crop = msk_raw[min(k * patch_size, h - patch_size):min((k + 1) * patch_size, h),
                       min(l * patch_size, w - patch_size): min((l + 1) * patch_size, w)].copy()
            #保存mask
            np.save(saveRootPath+"msk\\"+machine_name+"_"+rf"{idx:03d}.npy",msk_raw_crop)
    for i in range(1,101):
        cycName = "Cyc{:03d}".format(i)
        print(cycName)
        imgPathA = os.path.join(rootPath,dir_name,"res_deep_imageFilter\Image\Lane01",cycName,"R001C001_A.jpg")
        label_path = os.path.join(rootPath,dir_name,"res_deep_intent\Lane01\deepLearnData",cycName,"label\R001C001_label.npy")
        #middle的时候：
        image_middle = read4Img(imgPathA)
        label_middle = np.load(label_path)[np.newaxis,:]
        c, h, w = image_middle.shape

        #front:
        cycName_front = "Cyc{:03d}".format(i-1)
        imgPathA_front = os.path.join(rootPath,dir_name,"res_deep_imageFilter\Image\Lane01",cycName_front,"R001C001_A.jpg")
        label_front = os.path.join(rootPath,dir_name,"res_deep_intent\Lane01\deepLearnData",cycName_front,"label\R001C001_label.npy")
        try:
            image_front = read4Img(imgPathA_front)
            label_front = np.load(label_front)[np.newaxis,:]
        except:
            print(cycName_front)
            image_front = np.zeros((4, h, w))
            label_front = np.zeros((h,w))[np.newaxis,:]

        #behind
        cycName_behind = "Cyc{:03d}".format(i + 1)
        imgPathA_behind = os.path.join(rootPath, dir_name, "res_deep_imageFilter\Image\Lane01", cycName_behind,"R001C001_A.jpg")
        label_behind = os.path.join(rootPath, dir_name, "res_deep_intent\Lane01\deepLearnData", cycName_behind,"label\R001C001_label.npy")
        try:
            image_behind = read4Img(imgPathA_behind)
            label_behind = np.load(label_behind)[np.newaxis,:]
        except:
            print(cycName_behind)
            image_behind = np.zeros((4, h, w))
            label_behind = np.zeros((h, w))[np.newaxis,:]

        img = np.concatenate([image_front, image_middle, image_behind])
        label = np.concatenate([label_front, label_middle, label_behind])



        #剪裁：将一个图剪裁成

        patch_size = 256

        rows = int(h/patch_size)
        cols = int(w/patch_size)
        idx = 0
        for k in range(rows + 1):
            for l in range(cols):
                idx += 1
                img_crop = img[:,min(k * patch_size, h - patch_size):min((k + 1) * patch_size, h),min(l * patch_size, w - patch_size): min((l + 1) * patch_size, w)].copy()
                label_crop = label[:,min(k * patch_size, h - patch_size):min((k + 1) * patch_size, h),min(l * patch_size, w - patch_size): min((l + 1) * patch_size, w)].copy()
                np.save(saveRootPath+"img\\"+machine_name+"_"+rf"{i:03d}"+"_"+rf"{idx:03d}.npy",img_crop.astype(np.float32))
                np.save(saveRootPath + "label\\" + machine_name + "_"+rf"{i:03d}"+"_"+ rf"{idx:03d}.npy", label_crop(np.uint8))


# dirs = glob.glob(r"E:\data\resize_test\*")



