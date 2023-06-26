import cv2
import numpy as np
def invert_mapping2(map_x, map_y, img):
    h, w = map_x.shape
    inv_map_x = np.zeros_like(map_x)
    inv_map_y = np.zeros_like(map_y)

    for y in range(h):
        for x in range(w):
            inv_x, inv_y = map_x[y, x], map_y[y, x]
            if 0 <= inv_x < w and 0 <= inv_y < h:
                inv_map_x[y, x] = np.interp(inv_x, np.arange(w), img[y, :])
                inv_map_y[y, x] = np.interp(inv_y, np.arange(h), img[:, x])

    return inv_map_x, inv_map_y


def bilinear_interpolation(x, y, img):
    height, width = img.shape[:2]
    x1, y1 = int(x), int(y)
    x2, y2 = np.clip(x1 + 1, 0, width - 1), np.clip(y1 + 1, 0, height - 1)
    q11, q12 = img[y1, x1], img[y1, x2]
    q21, q22 = img[y2, x1], img[y2, x2]

    return q11 * (x2 - x) * (y2 - y) + q21 * (x - x1) * (y2 - y) + q12 * (x2 - x) * (y - y1) + q22 * (x - x1) * (y - y1)


def invert_mapping(map_x, map_y, img):
    h, w = map_x.shape
    inv_map_x = np.zeros_like(map_x)
    inv_map_y = np.zeros_like(map_y)

    for y in range(h):
        for x in range(w):
            inv_x, inv_y = map_x[y, x], map_y[y, x]
            if 0 <= inv_x < w and 0 <= inv_y < h:
                inv_map_x[y, x] = bilinear_interpolation(inv_x, inv_y, img)
                inv_map_y[y, x] = bilinear_interpolation(inv_x, inv_y, img)

    return inv_map_x, inv_map_y

def create_sample_image(size=(10, 10)):
    img = np.random.randint(0, 256, size, dtype=np.uint8)
    return img

def apply_mapping(img, map_x, map_y):
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

def create_random_mapping(shape):
    map_x = np.random.rand(*shape) * (shape[1] - 1)
    map_y = np.random.rand(*shape) * (shape[0] - 1)
    return map_x.astype(np.float32), map_y.astype(np.float32)
def create_custom_mapping(shape):
    map_x = np.zeros(shape, dtype=np.float32)
    map_y = np.zeros(shape, dtype=np.float32)

    for y in range(shape[0]):
        for x in range(shape[1]):
            if y % 2 == 0:
                # 偶数行
                map_x[y, x] = np.clip(x + 0.8, 0, shape[1] - 1)
                map_y[y, x] = np.clip(y + 0.8, 0, shape[0] - 1)
            else:
                # 奇数行
                map_x[y, x] = np.clip(x + 0.5, 0, shape[1] - 1)
                map_y[y, x] = np.clip(y + 0.5, 0, shape[0] - 1)

    return map_x, map_y
def main():
    # 创建一个 10x10 的随机图像
    img = cv2.imread("image.tif",0)

    # 创建随机映射
    map_x, map_y = create_custom_mapping(img.shape)

    # 应用映射变形图像
    warped_img = apply_mapping(img, map_x, map_y)

    # 计算逆映射
    inv_map_x, inv_map_y = invert_mapping(map_x, map_y, img)

    # 应用逆映射以恢复原始图像
    restored_img = apply_mapping(warped_img, inv_map_x, inv_map_y)

    # 保存原始图像、变形图像和恢复图像
    cv2.imwrite("original_image.png", img)
    cv2.imwrite("warped_image.png", warped_img)
    cv2.imwrite("restored_image.png", restored_img)

if __name__ == "__main__":
    main()