import numpy as np
def makeStrToKernel():


    str = "-6.299212598425196763e-02 -2.755905511811023167e-01 -6.299212598425196763e-02 -2.755905511811023167e-01 2.526418786692759433e+00 -2.755905511811023167e-01 -6.299212598425196763e-02 -2.755905511811023167e-01 -6.299212598425196763e-02"
    x0 = str.split(" ")[0]
    x1 = str.split(" ")[1]
    x2 = str.split(" ")[4]
    x0 = float(x0[:5])/(10 ** float(x0[-1]))
    x1 = float(x1[:5]) / (10 ** float(x1[-1]))
    x2 = float(x2[:5]) / (10 ** float(x2[-1]))

    kernel = np.array([[x0, x1, x0],
                       [x1, x2,x1],
                       [x0, x1, x0]])
    print(kernel)
    return kernel
makeStrToKernel()