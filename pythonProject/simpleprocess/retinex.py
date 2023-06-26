import cv2
import numpy as np

def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def SSR(img, size):
    L_blur = cv2.GaussianBlur(img, (size, size), 0)
    eps = float(1e-10)

    h, w = img.shape[:2]
    dst_img = np.zeros((h, w), dtype=np.float32)
    dst_Lblur = np.zeros((h, w), dtype=np.float32)
    dst_R = np.zeros((h, w), dtype=np.float32)

    img = replaceZeroes(img)
    L_blur = replaceZeroes(L_blur)
    img = img.astype(np.float32)
    L_blur = L_blur.astype(np.float32)
    cv2.log(img, dst_img)
    cv2.log(L_blur, dst_Lblur)
    log_R = cv2.subtract(dst_img, dst_Lblur)

    cv2.normalize(log_R, dst_R, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)

    minvalue, maxvalue, minloc, maxloc = cv2.minMaxLoc(log_R)
    for i in range(h):
        for j in range(w):
            log_R[i, j] = (log_R[i, j] - minvalue) * 255.0 / (maxvalue - minvalue)
    log_uint8 = cv2.convertScaleAbs(log_R)
    return log_uint8
img = cv2.imread("R001C001_A.tif")
r = SSR(img,31)
cv2.imshow("r",r)
cv2.waitKey(0)