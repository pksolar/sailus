import cv2
import numpy as np

# read the txt file containing the coordinates
typelist = ['A','C','G','T']
for type in typelist:
    name = f'R001C001_chanel_{type}'
    with open(rf'E:\data\resize_test\17_R1C78_resize_ori\res_for_reg\Lane01\sfile\{name}.btemp', 'r') as f:
        coordinates = f.readlines()[2:]

    # create a blank image with the specified dimensions
    img = np.zeros((2160, 4096), np.uint8)
    # img = cv2.imread("E:\code\python_PK\callbase\datasets\highDens_08\Image\Lane01\Cyc001\R001C001_T.tif")
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # iterate through the coordinates and draw white dots on the image
    for coord in coordinates:
        x, y = coord.strip().split(' ')
        x, y = round(float(x)), round(float(y))
        #cv2.circle(img, (x, y), 1, (255, 255, 255), -1)
        img[y,x] = 255
    cv2.imwrite(f"img/{name}.png",img)
# display the image
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img1 = cv2.imread("R001C001_chanel_A.jpg",0)
