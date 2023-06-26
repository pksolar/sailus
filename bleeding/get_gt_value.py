import numpy as np
import glob
import os
"""
分别获取ori 和final 4个通道的 100cycle gt均值
只做R001C001
每个cycle 和所有cycle的， 都需要。
"""






paths_final = glob.glob("E:\code\python_PK\callbase\datasets\highDens\Result\Lane01\deepLearnData_final\*\intensity\R001C001_A.npy")
paths_ori = glob.glob("E:\code\python_PK\callbase\datasets\highDens\Result\Lane01\deepLearnData_ori\*\intensity\R001C001_A.npy")
paths_label = glob.glob("E:\code\python_PK\callbase\datasets\highDens\Result\Lane01\deepLearnData_final\*\label\R001C001_label.npy")
msk = np.load("E:\code\python_PK\callbase\datasets\highDens\Result\Lane01\deepLearnData_final\R001C001_mask.npy").astype(int)

cycle = int(len(paths_final))
channel = 4
acgt = 4
every_cycle_sum_f = np.zeros((cycle, acgt, channel))  # record the sum of  every channls of each base in each cycle ,
every_cycle_num = np.zeros((cycle, channel))  # record the numbers of acgt in each cycle
every_cycle_mean_f = np.zeros((cycle, acgt, channel))

every_cycle_sum_o = np.zeros((cycle, acgt, channel))  # record the sum of  every channls of each base in each cycle ,
every_cycle_num = np.zeros((cycle, channel))  # record the numbers of acgt in each cycle
every_cycle_mean_o = np.zeros((cycle, acgt, channel))



for path_f,path_o,path_l in zip(paths_final,paths_ori,paths_label):


    list_A_f = np.zeros((4))  # get the sum of the four channle of A
    list_C_f = np.zeros((4))
    list_G_f = np.zeros((4))
    list_T_f = np.zeros((4))

    list_A_o = np.zeros((4))  # get the sum of the four channle of A
    list_C_o = np.zeros((4))
    list_G_o = np.zeros((4))
    list_T_o = np.zeros((4))

    num_A = 1
    num_C = 1
    num_G = 1
    num_T = 1

    dict_acgt_f = {1: list_A_f, 2: list_C_f, 3: list_G_f, 4: list_T_f}
    dict_acgt_o = {1: list_A_o, 2: list_C_o, 3: list_G_o, 4: list_T_o}
    dict_acgt_num = {1: num_A, 2: num_C, 3: num_G, 4: num_T}






    name = path_f.split("\\")[-3]
    name_num = int(name[-3:])
    print(name_num)
    #label contains 0,1,2,3,4,5 we only need 1,2,3,4 mean a c g t
    f_a_A = np.load(path_f)[np.newaxis,:]
    o_a_A = np.load(path_o)[np.newaxis,:]

    f_a_C = np.load(path_f.replace("_A","_C"))[np.newaxis,:]
    o_a_C = np.load(path_o.replace("_A","_C"))[np.newaxis,:]

    f_a_G = np.load(path_f.replace("_A","_G"))[np.newaxis,:]
    o_a_G = np.load(path_o.replace("_A","_G"))[np.newaxis,:]

    f_a_T = np.load(path_f.replace("_A","_T"))[np.newaxis,:]
    o_a_T = np.load(path_o.replace("_A","_T"))[np.newaxis,:]

    #concate the four channels
    f_a = np.concatenate([f_a_A,f_a_C,f_a_G,f_a_T])
    o_a = np.concatenate([o_a_A, o_a_C, o_a_G, o_a_T])


    l_a = np.load(path_l).astype(int)

    for i in range(msk.shape[0]):
        for j in range(msk.shape[1]):
            if msk[i,j] == 1: # only need the cluster mapped:
                if l_a[i,j] in [1,2,3,4]: #use the label to get the  value of the correspoding base
                    #print(f_a[:,i,j].shape)
                    #print( dict_acgt_f[l_a[i,j]].shape)
                    dict_acgt_f[l_a[i,j]] = dict_acgt_f[l_a[i,j]] + f_a[:,i,j]
                    dict_acgt_o[l_a[i, j]] = dict_acgt_o[l_a[i, j]] + o_a[:, i, j]
                    dict_acgt_num[l_a[i,j]] = dict_acgt_num[l_a[i,j]] + 1
                    #print("h")


    # a cycle is done,record the sum ,mean,and number
    mean_A_f = dict_acgt_f[1] / dict_acgt_num[1]
    mean_C_f = dict_acgt_f[2] / dict_acgt_num[2]
    mean_G_f = dict_acgt_f[3] / dict_acgt_num[3]
    mean_T_f = dict_acgt_f[4] / dict_acgt_num[4]

    mean_A_o = dict_acgt_o[1] / dict_acgt_num[1]
    mean_C_o = dict_acgt_o[2] / dict_acgt_num[2]
    mean_G_o = dict_acgt_o[3] / dict_acgt_num[3]
    mean_T_o = dict_acgt_o[4] / dict_acgt_num[4]


    num_a = np.array([dict_acgt_num[1],dict_acgt_num[2],dict_acgt_num[3],dict_acgt_num[4]])
    mean_f = np.array([mean_A_f,mean_C_f,mean_G_f,mean_T_f])
    mean_o = np.array([mean_A_o, mean_C_o, mean_G_o, mean_T_o])

    sum_f = np.array([dict_acgt_f[1],dict_acgt_f[2],dict_acgt_f[3],dict_acgt_f[4]])
    sum_o = np.array([dict_acgt_o[1],dict_acgt_o[2],dict_acgt_o[3],dict_acgt_o[4]])


    every_cycle_mean_f[name_num-1] = mean_f
    every_cycle_mean_o[name_num - 1] = mean_o
    every_cycle_num[name_num-1] = num_a

    every_cycle_sum_f[name_num-1] = sum_f
    every_cycle_sum_o[name_num-1] = sum_o
    #np.save("gtvalue/{}.npy".format(name),mean_f)
    #对其中一些进行清0


np.save("gtvalue/mean_f.npy", every_cycle_mean_f)
np.save("gtvalue/mean_o.npy", every_cycle_mean_o)
np.save("gtvalue/cycle_number.npy", every_cycle_num)

np.save("gtvalue/sum_f.npy", every_cycle_sum_f)
np.save("gtvalue/sum_o.npy", every_cycle_sum_o)

print(mean_T_f)






