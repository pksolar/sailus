import numpy as np
from scipy.signal import convolve2d
import cv2

inputs = cv2.imread("R001C001_A.tif",0)


conv_weights = np.array([[-0.1,-0.2,-0.1],[-0.2,1.2,-0.2],[-0.1,-0.2,-0.1]])
conv_weights2 = np.array([[-0.0719, -0.0202, -0.0816],
          [-0.0125,  0.4151,  0.0040],
          [-0.0803, -0.0035, -0.0911]])
conv_output = convolve2d(inputs,conv_weights, mode='same', boundary='fill')
dst = cv2.filter2D(inputs,-1,conv_weights,)
print(dst.shape)
# cv2.imwrite("result.tif",dst)
cv2.imwrite("ori.jpg",inputs[1000:1512,1000:1512])
cv2.imwrite("result.jpg",dst[1000:1512,1000:1512])
#cv2.imshow("Average filtered", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()