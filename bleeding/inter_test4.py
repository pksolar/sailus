import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def bilinear_interpolate(source, input_pos, scale=2, pad=1):
    """
    :param source:
    :param input_pos: 维度： 2 x  30w (点的个数)  第一个维度中0：x坐标，（h），1：y坐标 (w)
    :param scale:
    :param pad:
    :return:
    """

    sour_shape = source.shape
    (sh, sw) = (sour_shape[-2], sour_shape[-1])
    padding = pad * np.ones((sour_shape[0], sour_shape[1], sh + 1, sw + 1))
    padding[:, :, :-1, :-1] = source
    # 目标图像h,w
    (th, tw) = (round(scale * sh), round(scale * sw))
    # 生成grid,新图，存放 新图在老图上对应的坐标
    grid = np.array(np.meshgrid(np.arange(th), np.arange(tw)), dtype=np.float32)  # 2,59,47
    xy = np.copy(grid)
    # 计算新图到老图上的坐标。
    xy[0] = sh / th * (xy[0] + 0.5) - 0.5
    xy[1] = sw / tw * (xy[1] + 0.5) - 0.5

    x = input_pos[0]
    y = input_pos[1]

    # 拉平，这里和我的数据十分相似了。里面是对应的坐标
    x = xy[0].flatten()
    y = xy[1].flatten()

    # 计算取整的坐标，并拉平
    clip = np.floor(xy).astype(np.int)
    cx = clip[0].flatten()
    cy = clip[1].flatten()

    f1 = padding[:, :, cx, cy]
    f2 = padding[:, :, cx + 1, cy]
    f3 = padding[:, :, cx, cy + 1]
    f4 = padding[:, :, cx + 1, cy + 1]

    a = cx + 1 - x
    b = x - cx
    c = cy + 1 - y
    d = y - cy

    fx1 = a * f1 + b * f2
    fx2 = a * f3 + b * f4
    fy = c * fx1 + d * fx2
    fy = fy.reshape(fy.shape[0], fy.shape[1], tw, th).transpose((0, 1, 3, 2))
    return fy


if __name__ == '__main__':
    path_list = ['inter/dog.jpg']
    imgs = []
    for path in path_list:
        im = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB) / 255
        imgs.append(im)
    imgs = np.array(imgs).transpose((0, 3, 1, 2))
    interps_0d1 = bilinear_interpolate(imgs, scale=0.1)
    interps_2d2 = bilinear_interpolate(imgs, scale=2.2)
    for im, interp0, interp1 in zip(imgs, interps_0d1, interps_2d2):
        plt.figure()
        plt.subplot(131)
        plt.imshow(im.transpose(1, 2, 0))
        plt.subplot(132)
        plt.imshow(interp0.transpose(1, 2, 0))
        plt.title('scale to 0.1 times of the original image')
        plt.subplot(133)
        plt.imshow(interp1.transpose(1, 2, 0))
        plt.title('scale to 2.2 times of the original image')
        plt.show()