import numpy as np
import cv2

def RGB2HSV(src):
    row, col, _ = src.shape
    dst = np.zeros((row, col, 3), dtype=np.float32)
    for i in range(row):
        for j in range(col):
            b = src[i, j, 0] / 255.0
            g = src[i, j, 1] / 255.0
            r = src[i, j, 2] / 255.0
            minn = min(r, min(g, b))
            maxx = max(r, max(g, b))
            dst[i, j, 2] = maxx  # V
            delta = maxx - minn
            if maxx != 0:
                s = delta / maxx
            else:
                s = 0
            if r == maxx:
                h = (g - b) / delta
            elif g == maxx:
                h = 2 + (b - r) / delta
            else:
                h = 4 + (r - g) / delta
            h *= 60
            if h < 0:
                h += 360
            dst[i, j, 0] = h
            dst[i, j, 1] = s
    return dst

def HSV2RGB(src):
    row, col, _ = src.shape
    dst = np.zeros((row, col, 3), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            h = src[i, j, 0]
            s = src[i, j, 1]
            v = src[i, j, 2]
            if s == 0:
                r = g = b = v
            else:
                h /= 60
                offset = int(np.floor(h))
                f = h - offset
                p = v * (1 - s)
                q = v * (1 - s * f)
                t = v * (1 - s * (1 - f))
                if offset == 0:
                    r, g, b = v, t, p
                elif offset == 1:
                    r, g, b = q, v, p
                elif offset == 2:
                    r, g, b = p, v, t
                elif offset == 3:
                    r, g, b = p, q, v
                elif offset == 4:
                    r, g, b = t, p, v
                elif offset == 5:
                    r, g, b = v, p, q
            dst[i, j, 0] = int(b * 255)
            dst[i, j, 1] = int(g * 255)
            dst[i, j, 2] = int(r * 255)
    return dst

def work(src):
    row, col, _ = src.shape
    # now = RGB2HSV(src)
    now = src.copy()
    H = np.zeros((row, col), dtype=np.float32)
    S = np.zeros((row, col), dtype=np.float32)
    V = np.zeros((row, col), dtype=np.float32)
    H[:,:] = now[:,:,0]
    S[:, :] = now[:, :, 1]
    V[:, :] = now[:, :, 2]
    # for i in range(row):
    #     for j in range(col):
    #         H[i, j] = now[i, j, 0]
    #         S[i, j] = now[i, j, 1]
    #         V[i, j] = now[i, j, 2]
    kernel_size = min(row, col)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    SIGMA1 = 15
    SIGMA2 = 80
    SIGMA3 = 250
    q = np.sqrt(2.0)
    F = np.zeros((row, col), dtype=np.float32)
    F1 = cv2.GaussianBlur(V, (kernel_size, kernel_size), SIGMA1 / q)
    F2 = cv2.GaussianBlur(V, (kernel_size, kernel_size), SIGMA2 / q)
    F3 = cv2.GaussianBlur(V, (kernel_size, kernel_size), SIGMA3 / q)
    for i in range(row):
        for j in range(col):
            F[i, j] = (F1[i, j] + F2[i, j] + F3[i, j]) / 3.0
    average = np.mean(F)
    out = np.zeros((row, col), dtype=np.float32)
    for i in range(row):
        for j in range(col):
            gamma = np.power(0.5, (average - F[i, j]) / average)
            out[i, j] = np.power(V[i, j], gamma)
    merge_ = cv2.merge((H, S, out))
    dst = merge_.copy()
    return dst

src = cv2.imread(r"E:\data\resize_test\22_resize_ori\Lane01\Cyc001\R001C001_A.tif")
dst = work(src).astype(np.uint8)

cv2.imwrite("dst_uevne.tif",dst)
cv2.imshow("dst",dst)
cv2.waitKey(0)