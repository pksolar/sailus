import cv2
import numpy as np
# read in two images
img1 = cv2.imread('img/2.tif')
img2 = cv2.imread('img/3.tif')

# get the center 512x512 region of each image
h, w = img1.shape[:2]
# center1 = img1[h//2-256:h//2+256, w//2-256:w//2+256]
h, w = img2.shape[:2]
# center2 = img2[h//2-256:h//2+256, w//2-256:w//2+256]

# convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
# find the x and y translation using cv2.phaseCorrelate
dx, dy = cv2.phaseCorrelate(gray1, gray2)

# apply the translation to img2
rows, cols = img2.shape[:2]
M = np.float32([[1, 0, dx[1]], [0, 1, dx[0]]])
img2_translated = cv2.warpAffine(img2, M, (cols, rows))
img2_translated = cv2.cvtColor(img2_translated, cv2.COLOR_BGR2GRAY)
cv2.imwrite("img/result_inv_full.tif",img2_translated)
# display the original and translated images side by side
# cv2.imshow('Original Image 1', center1)
# cv2.imshow('Original Image 2', center2)
# cv2.imshow('Translated Image 2', img2_translated)
cv2.waitKey(0)
