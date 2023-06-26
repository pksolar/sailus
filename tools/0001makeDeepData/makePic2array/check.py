import cv2
import numpy as np
a = np.load(r"E:\data\deepData\train\img\08.1h_009.npy")
img = cv2.imread("source/R001C001_A.jpg",0)
f = np.save("source/ori.npy",img)
g = np.save("source/float32.npy",img.astype(np.float32))
h = np.save("source/float16.npy",img.astype(np.float16))
h = np.save("source/int.npy",img.astype(int))
h = np.save("source/uint8.npy",img.astype(np.uint8))
h = np.save("source/float64.npy",img.astype(np.float64))
# h,w = img.shape
# patch_size = 256
# patch_size = 256
# rows = int(h/patch_size)
# cols = int(w/patch_size)
# idx = 0
# for k in range(rows+1):
#     for l in range(cols):
#         idx += 1
#         print(l)
#         crop = img[ min(k*patch_size,h-patch_size):min((k+1)*patch_size,h), min(l*patch_size,w-patch_size): min((l+1)*patch_size,w)].copy()
#         cv2.imwrite(rf"source/{idx:05d}.jpg",crop)