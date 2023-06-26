import numpy as np
def Q99(img):
    """

    :param img: array 矩阵
    :return:
    """
    a,b = img.shape
    img_ = img.reshape(1,a*b)
    img_ = img_[0,:]
    print(img_.shape)
    print(img_)
    img_list = img_.tolist()
    print(img_list)
    g = img_list.remove(0)
    print(len(g))

a = np.zeros((5,5))
b = Q99(a)
