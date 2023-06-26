import numpy as np
import glob
import cv2
import os
"""
A: Parameter containing:
tensor([[[[0.0748, 0.2117, 0.1461],
          [0.2361, 0.0905, 0.2455],
          [0.2173, 0.2030, 0.0019]]]], device='cuda:0', requires_grad=True)  -> Parameter containing:
tensor([0.2507], device='cuda:0', requires_grad=True) 
C: Parameter containing:
tensor([[[[ 0.0982, -0.0637, -0.0046],
          [ 0.0789,  0.0405,  0.0772],
          [-0.0647,  0.0714, -0.0632]]]], device='cuda:0', requires_grad=True)  -> Parameter containing:
tensor([0.1185], device='cuda:0', requires_grad=True) 
G: Parameter containing:
tensor([[[[-0.0146,  0.1590,  0.0952],
          [ 0.1045, -0.0299,  0.1624],
          [ 0.1277,  0.0100,  0.1733]]]], device='cuda:0', requires_grad=True)  -> Parameter containing:
tensor([0.5070], device='cuda:0', requires_grad=True) 
T Parameter containing:
tensor([[[[ 0.2080, -0.0964, -0.1035],
          [ 0.2200,  0.1508,  0.2223],
          [-0.0691,  0.1943,  0.2238]]]], device='cuda:0', requires_grad=True)  -> Parameter containing:
tensor([0.5725], device='cuda:0', requires_grad=True)


"""

kernela = np.array([[0.0748, 0.2117, 0.1461],
          [0.2361, 0.0905, 0.2455],
          [0.2173, 0.2030, 0.0019]])
kernelc= np.array([[ 0.0982, -0.0637, -0.0046],
          [ 0.0789,  0.0405,  0.0772],
          [-0.0647,  0.0714, -0.0632]])
kernelg= np.array([[-0.0146,  0.1590,  0.0952],
          [ 0.1045, -0.0299,  0.1624],
          [ 0.1277,  0.0100,  0.1733]])
kernelt= np.array([[ 0.2080, -0.0964, -0.1035],
          [ 0.2200,  0.1508,  0.2223],
          [-0.0691,  0.1943,  0.2238]])


paths_x = glob.glob("E:\code\python_PK\callbase\datasets\highDens\Image\Lane01\*\R001C001_A.tif")
for path in paths_x:
    name = path.split("\\")[-2]
    aA = cv2.imread(path,0)
    aC = cv2.imread(path.replace("_A","_C"),0)
    aG = cv2.imread(path.replace("_A","_G"),0)
    aT = cv2.imread(path.replace("_A","_T"),0)

    outA = cv2.filter2D(aA,-1,kernela)
    outC = cv2.filter2D(aC, -1, kernelc)
    outG = cv2.filter2D(aG, -1, kernelg)
    outT = cv2.filter2D(aT, -1, kernelt)
    if not os.path.exists("resultimg/Lane01/{}".format(name)):
        os.mkdir("resultimg/Lane01/{}".format(name))
    cv2.imwrite("resultimg/Lane01/{}/R001C001_A.tif".format(name),outA)
    cv2.imwrite("resultimg/Lane01/{}/R001C001_C.tif".format(name), outC)
    cv2.imwrite("resultimg/Lane01/{}/R001C001_G.tif".format(name), outG)
    cv2.imwrite("resultimg/Lane01/{}/R001C001_T.tif".format(name), outT)


