import cv2
import os

# set path to folder containing images
path = r"E:\code\python_PK\VoxelMorph-torch-master\reg\fusionimg"

# loop through all files in folder
for filename in os.listdir(path):
    # check if file is an image
    if filename.endswith(".tif") or filename.endswith(".png"):
        # read image
        img = cv2.imread(os.path.join(path, filename),0)
        # normalize image to range 0-255
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        # save normalized image

        cv2.imwrite(os.path.join(path, filename.replace(".","norm.")), img)
