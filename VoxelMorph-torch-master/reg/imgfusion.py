import cv2
import glob
import numpy as np
typelist = ["A","C","G","T"]
for type_ in typelist:
    paths = glob.glob(rf"E:\code\python_PK\VoxelMorph-torch-master\reg\phase_imgRound_17_R1C78_resize_ori\Lane01\*\R001C001_{type_}.tif")
    n = len(paths)
    # read in n images
    images = []
    for path in paths:
        img = cv2.imread(path,0)
        images.append(img)

    # calculate weight for each image
    weight = 1/n

    # initialize sum image
    sum_img = images[0] * weight

    # add weighted images to sum image
    for i in range(1, n):
        sum_img += images[i] * weight
        if i >20:
            break

    sum_img = 255*(sum_img-np.max(sum_img))/(np.max(sum_img)-np.min(sum_img))
    cv2.imwrite(rf"fusionimg/20_round_{type_}.tif",sum_img.astype(np.uint8))

    # display sum image
    # cv2.imshow("Sum Image", sum_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
