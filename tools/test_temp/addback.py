import  numpy as np
import  cv2
import glob
import os
save_root_dir = "img"
paths = glob.glob(r"E:\data\resize_test\08_resize_ori\res_deep\Image\Lane01/*/R001C001_A.jpg")
for path in paths:
    name_path_C = path.replace("_A", '_C')
    name_path_G = path.replace("_A", '_G')
    name_path_T = path.replace("_A", '_T')

    imgA = cv2.imread(path, 0) + 1
    imgC = cv2.imread(name_path_C, 0) + 1
    imgG = cv2.imread(name_path_G, 0) + 1
    imgT = cv2.imread(name_path_T, 0) + 1

    cycname = path.split("\\")[-2]
    os.makedirs(f"{save_root_dir}/Lane01/{cycname}",exist_ok=True)
    cv2.imwrite(f"{save_root_dir}/Lane01/{cycname}/R001C001_A.tif", imgA)
    cv2.imwrite(f"{save_root_dir}/Lane01/{cycname}/R001C001_C.tif", imgC)
    cv2.imwrite(f"{save_root_dir}/Lane01/{cycname}/R001C001_G.tif", imgG)
    cv2.imwrite(f"{save_root_dir}/Lane01/{cycname}/R001C001_T.tif", imgT)
