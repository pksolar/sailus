import numpy as np
import cv2
import glob
import os
#E:\code\python_PK\callbase\datasets\highdens_22_ori\Lane01\*\R001C001_A.tif
#r"E:\code\python_PK\callbase\datasets\highDens_08\Image\Lane01\*\R001C001_A.tif",
#r"E:\code\python_PK\callbase\datasets\highdens_22_ori\Lane01\*\R001C001_A.tif",

"""

h_1.2: 2592.0
w_1.2: 4915.2
--------------
h_1.25: 2700.0
w_1.25: 5120.0
--------------
h_1.3: 2808.0
w_1.3: 5324.8
--------------
h_1.35: 2916.0
w_1.35: 5529.6
--------------
h_1.4: 3024.0
w_1.4: 5734.4
--------------

Process finished with exit code 0



"""
pathlist = [r"E:\code\python_PK\bleeding\img_process\1.9.39_data\Lane01\*\*_A.tif",
            r"E:\code\python_PK\bleeding\img_process\1.9.1_data\Lane01\*\*_A.tif"]
for path in pathlist:
    save_dir_name = path.split("\\")[-4].replace("_data","")
    paths_x = glob.glob(path)
    dict_resize = {1.2:[2592,4914],
                   1.25:[2700,5120],
                   1.3:[2808,4324],
                   1.35:[2916,5530],
                   1.4:[3024,5734]}
    for key in dict_resize:

        save_dir = save_dir_name + f"_resize{key}"

        i= 0
        height = dict_resize[key][0]
        width = dict_resize[key][1]


        for path in paths_x:
            name = path.split("\\")[-2]
            imgname = path.split("\\")[-1][:8]
            aA = cv2.imread(path, 0)
            aA = cv2.resize(aA, (width, height))
            aC = cv2.imread(path.replace("_A", "_C"), 0)
            aC = cv2.resize(aC, (width, height))
            aG = cv2.imread(path.replace("_A", "_G"), 0)
            aG = cv2.resize(aG, (width, height))
            aT = cv2.imread(path.replace("_A", "_T"), 0)
            aT = cv2.resize(aT,(width, height))
            print(imgname)

            if not os.path.exists("{}/Lane01/{}".format(save_dir, name)):
                os.makedirs("{}/Lane01/{}".format(save_dir, name))
            cv2.imwrite("{}/Lane01/{}/{}_A.tif".format(save_dir, name,imgname), aA)
            cv2.imwrite("{}/Lane01/{}/{}_C.tif".format(save_dir, name,imgname), aC)
            cv2.imwrite("{}/Lane01/{}/{}_G.tif".format(save_dir, name,imgname), aG)
            cv2.imwrite("{}/Lane01/{}/{}_T.tif".format(save_dir, name,imgname), aT)
            i += 1
            print(i)
