import cv2
import numpy as np
b = np.load("test_intensity.npy")
print(b.shape)
b = b[1:300,1:300]
print(b.shape)