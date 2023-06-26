import os

import cv2
import numpy as np
import glob

# read in two images
#E:\data\resize_test\17_R1C78_resize_ori\Lane01
def reg(img1,img2,cycname,machine_name,imgname):
    h, w = img1.shape[:2]
    center1 = img1[h//2-256:h//2+256, w//2-256:w//2+256].astype(np.float32)
    h, w = img2.shape[:2]
    center2 = img2[h//2-256:h//2+256, w//2-256:w//2+256].astype(np.float32)

    # convert to grayscale


    # center1 = cv2.cvtColor(center1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # center2 = cv2.cvtColor(center2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # find the x and y translation using cv2.phaseCorrelate
    dx, dy = cv2.phaseCorrelate(center1, center2)

    # apply the translation to img2
    rows, cols = img2.shape[:2]
    #M = np.float32([[1, 0, dx[1]], [0, 1, dx[0]]])
    M = np.float32([[1, 0, round(dx[1])], [0, 1, round(dx[0])]])
    # M = np.float32([[1, 0, 0], [0, 1, 0]])
    img2_translated = cv2.warpAffine(img2, M, (cols, rows))
    #img2_translated = cv2.cvtColor(img2_translated, cv2.COLOR_BGR2GRAY)
    cycdirname = rf"E:\code\python_PK\VoxelMorph-torch-master\reg\phase_imgRound_{machine_name}\Lane01\{cycname}"
    if not os.path.exists(cycdirname):
        os.makedirs(cycdirname)
    cv2.imwrite(os.path.join(cycdirname,imgname), img2_translated)

name_machine = "17_R1C78_resize_ori"
tempA =rf"E:\data\resize_test\{name_machine}\Lane01\Cyc001\R001C001_A.tif"
tempC =rf"E:\data\resize_test\{name_machine}\Lane01\Cyc001\R001C001_C.tif"
tempG =rf"E:\data\resize_test\{name_machine}\Lane01\Cyc001\R001C001_G.tif"
tempT =rf"E:\data\resize_test\{name_machine}\Lane01\Cyc001\R001C001_T.tif"

tempAimg = cv2.imread(tempA,0)
tempCimg = cv2.imread(tempC,0)
tempGimg = cv2.imread(tempG,0)
tempTimg = cv2.imread(tempT,0)
templist = [tempAimg,tempCimg,tempGimg,tempTimg]
dictacgt = {1:"R001C001_A.tif",2:"R001C001_C.tif",3:"R001C001_G.tif",4:"R001C001_T.tif"}
paths = glob.glob(fr"E:\data\resize_test\{name_machine}\Lane01\*\R001C001_A.tif")
for path in paths:
    cycname = path.split("\\")[-2]

    # if  cycname == "Cyc001":
    #     continue
    imgA = cv2.imread(path,0)
    imgC = cv2.imread(path.replace("A", "C"),0)
    imgG = cv2.imread(path.replace("A", "G"),0)
    imgT = cv2.imread(path.replace("A", "T"),0)
    listimg = [imgA,imgC,imgG,imgT]
    for i,(img1,img2) in enumerate(zip(templist,listimg)):
        #imgname = path.split("\\")[-1]
        reg(img1,img2,cycname,name_machine,dictacgt[i+1])


# get the center 512x512 region of each image

# display the original and translated images side by side
# cv2.imshow('Original Image 1', center1)
# cv2.imshow('Original Image 2', center2)
# cv2.imshow('Translated Image 2', img2_translated)
cv2.waitKey(0)
