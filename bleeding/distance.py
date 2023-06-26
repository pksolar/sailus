import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('imgtest/testT.tif', cv.IMREAD_GRAYSCALE)
_ret, img2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
opn = cv.morphologyEx(img2, cv.MORPH_OPEN, kernel)
distance = cv.distanceTransform(opn, cv.DIST_L2, 3)
_ret, result = cv.threshold(distance, 0.7 * distance.max(), 255, cv.THRESH_BINARY)
cv.imshow("dd",distance)
cv.waitKey(0)
