import cv2
import numpy as np
import cv2

image = cv2.imread("img/R001C001_A.tif")

gamma =1.25
corrected_image = ((image / 180.0) ** gamma) * 180.0
corrected_image = corrected_image.astype("uint8")

cv2.imwrite("corrected_image_A.tif", corrected_image)
