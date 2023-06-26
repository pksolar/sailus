import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def bilinear_interpolate(source, scale=2, pad=0.5):
    sour_shape = source.shape
    (sh, sw) = (sour_shape[-2], sour_shape[-1])
    padding = pad * np.ones((sour_shape[0], sour_shape[1], sh + 1, sw + 1))
    padding[:, :, :-1, :-1] = source

    (th, tw) = (round(scale * sh), round(scale * sw))

    grid = np.array(np.meshgrid(np.arange(th), np.arange(tw)), dtype=np.float32)
    xy = np.copy(grid)

    xy[0] = sh / th * (xy[0]+0.5) - 0.5
    xy[1] = sw / tw * (xy[1]+0.5) - 0.5

    x = xy[0].flatten()
    y = xy[1].flatten()

    clip = np.floor(xy).astype(np.int)
    cx = clip[0].flatten()
    cy = clip[1].flatten()

    #cx,cy 是具体的值，所以f1也是具体的值，但是由于cx和cy是array，所以，F1是array
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