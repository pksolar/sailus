import cv2
import numpy as np
import glob
import os
def moveA(img,cycname,imgname):
    cycdirname = rf"E:\code\python_PK\VoxelMorph-torch-master\reg\deformed_img_elaxtic\Lane01\{cycname}"
    if not os.path.exists(cycdirname):
        os.makedirs(cycdirname)
    cv2.imwrite(os.path.join(cycdirname, imgname), img)


def deform(img,deformation_field,cycname,imgname):
    # Get image A dimensions
    height, width = img.shape

    # Create meshgrid for x and y coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Add deformation field to meshgrid
    x_deformed = x + deformation_field[0]
    y_deformed = y + deformation_field[1]

    # Remap image A using deformed meshgrid
    img_deformed = cv2.remap(img, x_deformed.astype(np.float32), y_deformed.astype(np.float32),
                               cv2.INTER_LINEAR).astype(np.uint8)
    # Display deformed image A
    cycdirname = rf"E:\code\python_PK\VoxelMorph-torch-master\reg\deformed_img_elaxtic\Lane01\{cycname}"
    if not os.path.exists(cycdirname):
        os.makedirs(cycdirname)
    cv2.imwrite(os.path.join(cycdirname, imgname), img_deformed)

# Load deformation field data
rootdir =rf"E:\code\python_PK\VoxelMorph-torch-master\reg_elastic\reg_example"
deformation_field_C = np.load(rf'{rootdir}\flow_20_Cnorm.npy')
deformation_field_G = np.load(rf'{rootdir}\flow_20_Gnorm.npy')
deformation_field_T = np.load(rf'{rootdir}\flow_20_Tnorm.npy')

paths = glob.glob(r"E:\code\python_PK\VoxelMorph-torch-master\reg\phase_imgRound_17_R1C78_resize_ori\Lane01\*\R001C001_A.tif")
for path in paths:

    # Load image A
    imgA = cv2.imread(path,0)
    imgC = cv2.imread(path.replace("_A", "_C"),0)
    imgG = cv2.imread(path.replace("_A", "_G"),0)
    imgT = cv2.imread(path.replace("_A", "_T"),0)
    # img_C = cv2.imread(paths,0)
    cycname = path.split("\\")[-2]
    moveA(imgA,cycname,"R001C001_A.tif")
    deform(imgC,deformation_field_C,cycname,"R001C001_C.tif")
    deform(imgG, deformation_field_G,cycname,"R001C001_G.tif")
    deform(imgT, deformation_field_T,cycname,"R001C001_T.tif")

    # Get image A dimensions


