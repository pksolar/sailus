import numpy as np
import glob
"""
get every cycle gt img
get all the mean of the img

"""
import numpy as np
mean_f = np.load("gtvalue/mean_o.npy") #  cycle:101 ,acgct:4 , 4channel vlaue:4
sum_f = np.load("gtvalue/sum_o.npy")  #  cycle:101 ,acgct:4 , 4channel vlaue:4
number = np.load("gtvalue/cycle_number.npy") #cycle:101, acgt_number:4
#
#
# mean_o = np.load("gtvalue/mean_o.npy") #  cycle:101 ,acgct:4 , 4channel vlaue:4
# sum_o = np.load("gtvalue/sum_o.npy")  #  cycle:101 ,acgct:4 , 4channel vlaue:4
#
#
# sum_o_all = np.sum(sum_o,axis=0)
# number_all = np.sum(number,axis=0)
#
# # print("axis =0:\n",sum_f_all)
# print("axis =0:\n",sum_o_all/number_all)

paths_label = glob.glob("E:\code\python_PK\callbase\datasets\highDens\Result\Lane01\deepLearnData_final\*\label\R001C001_label.npy")
msk = np.load("E:\code\python_PK\callbase\datasets\highDens\Result\Lane01\deepLearnData_final\R001C001_mask.npy").astype(int)
for path in paths_label:
    temp_ = np.zeros((2160, 4096, 4))
    name = path.split("\\")[-3]
    name_num = int(name[-3:])
    print("name:",name_num)
    acgtvalue = mean_f[name_num-1] # 取出当前cycyle的acgt：4x4的array
    label = np.load(path).astype(int)
    for j in range(label.shape[0]):
        for k in range(label.shape[1]):
            if label[j,k] in [1,2,3,4]:
                temp_[j,k,:] = acgtvalue[label[j,k]-1]
    np.save("gtvalue/gtimg_ori/{}.npy".format(name),temp_)







