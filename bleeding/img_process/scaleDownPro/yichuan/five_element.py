# from skimage.filters import convolve
from sko.GA import GA
import numpy as np
import  cv2
import glob
import os
paths_A = glob.glob(r"E:\data\resize_test\08_resize_ori_for_yichuan\Lane01\*\R001C001_A.tif")
# 17_R1C78_resize_oriE:\data\resize_test\17_R1C78_resize1.25
import threading
import autoQ30mapping

"""

    08号机ori：88.75
  ori 
  kernel = np.array([[0.02134374, 0.08381951, 0.02134374],
                       [0.16381951, -1.41934703, 0.16381951],
                       [0.02134374, 0.16381951, 0.02134374]])  mapping  89.6

                       kernel = np.array([[0.04134374, 0.32381951, 0.04134374],
                       [0.32381951, -2.51934703, 0.32381951],
                       [0.04134374, 0.32381951, 0.04134374]]) # mapping 89.91




"""

def f1(params):
    return 0.5-(4*params[0] + 8* params[1] + 4* params[2]+4*params[3] + 4*params[4] + params[5])
def f2(params):
    return abs(params[0]) - abs(params[3])
def f3(params):
    return abs(params[2]) - abs(params[4])
def f4(params):
    return abs(params[1]) - abs(params[4])
# def constraint_func2(params):
#
#     return 0.2-(params[0] + params[1] + params[2])

def convolve(img_path,kernel, path_save, str_,imgnameA):
    # kernel = np.array([[0.1134374, 0.22381951, 0.1134374],
    #                    [0.22381951, -2.51934703, 0.22381951],
    #                    [0.1134374, 0.22381951, 0.1134374]])
    img = cv2.imread(img_path, 0)
    dst = cv2.filter2D(img, -1, kernel).astype(np.uint8)
    # dst = cv2.GaussianBlur(dst, (3, 3), 0.9).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(path_save, imgnameA.replace("_A", str_)), dst)
    # return dst


def evaluate_kernel(params):

    kernel =np.array( [[params[0], params[1], params[2], params[1], params[0]],
                       [params[1], params[3], params[4], params[3], params[1]],
                       [params[2], params[4], params[5], params[4], params[2]],
                       [params[1], params[3], params[4], params[3], params[1]],
                       [params[0], params[1], params[2], params[1], params[0]]])
    with open('kernel_mapping_5.txt', 'a') as f:
        np.savetxt(f, kernel)
    #对图像采用多线程卷积操作：
    threads = []

    for pathA in paths_A:
        imgnameA = pathA.split("\\")[-1].replace(pathA.split("\\")[-1].split(".")[0], "R001C001_A")
        cycname = pathA.split("\\")[-2]
        print(cycname, " ", imgnameA)
        path_save = r"E:\code\python_PK\bleeding\img_process\scaleDownPro\yichuan_5x5\image\Lane01\{}".format(cycname)
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        # create threads for each channel
        threadA = threading.Thread(target=convolve, args=(pathA,kernel, path_save, "_A",imgnameA))
        threadC = threading.Thread(target=convolve, args=(pathA.replace("R001C001_A", "R001C001_C"),kernel, path_save, "_C",imgnameA))
        threadG = threading.Thread(target=convolve, args=(pathA.replace("R001C001_A", "R001C001_G"),kernel, path_save, "_G",imgnameA))
        threadT = threading.Thread(target=convolve, args=(pathA.replace("R001C001_A", "R001C001_T"),kernel, path_save, "_T",imgnameA))

        # start threads
        threadA.start()
        threadC.start()
        threadG.start()
        threadT.start()

        # add threads to the list
        threads.append(threadA)
        threads.append(threadC)
        threads.append(threadG)
        threads.append(threadT)

    # wait for all threads to finish
    for thread in threads:
        thread.join()

    #图像做完，autoq30,automapping:
    rootdir = r"E:\code\python_PK\bleeding\img_process\scaleDownPro\yichuan_5x5\image\\"
    resdir = "res"

    mappingrate,mappedreads = autoQ30mapping.main(rootdir,resdir,cycleNum=30)

    print( "mapping: "+str(mappingrate)+",mapped Reads: "+str(mappedreads))
    print("kernel:",kernel)
    print("score:")

    # 在这里使用卷积核对图像进行操作，并计算评分结果
    # 这里假设你有一个名为score_function的函数来计算评分，你需要将其替换为你实际的评分函数

    # processed_image = convolve(image, kernel)
    # score = score_function(processed_image)
    score = mappingrate + mappedreads / 15000
    print("score:",score)

    with open('kernel_mapping_5.txt', 'a') as f:
        f.write("mapping: " + str(mappingrate) + "mapped Reads: " + str(mappedreads) +"score: "+str(score) + '\n')
        f.write("------------------------------------------------------------------------" + '\n')

    return -score  # 返回负分数，因为遗传算法默认最小化目标函数


# param_bound = [(lower_bound, upper_bound) for _ in range(3)]  # 每个参数的取值范围
ga = GA(func=evaluate_kernel, n_dim=6, size_pop=60, prob_mut=0.05,max_iter=100, lb=[-0.4,-0.7,-1,-1 ,-1, 2], ub=[0, 0,0,0,0, 20], constraint_ueq=[f1,f2,f3,f4], precision=0.01)
best_params, best_score = ga.run()
print(best_params,best_score)
with open('kernel_mapping_5.txt', 'a') as f:
    f.write(f"{best_score},{best_params}")
