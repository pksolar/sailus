import numpy as np
right = np.load(r"E:\data\resize_test\08_resize_ori\res_deep\Lane01\deepLearnData\Cyc010\label\R001C001_label.npy")
right_same = right.copy()
right_same[right_same == 0] = 10
raw = np.load(r"E:\data\resize_test\08_resize_ori\res\Lane01\deepLearnData\Cyc010\label\R001C001_label.npy")
raw_same = raw.copy()
raw_same[raw_same == 0] = 20
diff = np.where(right != raw )
same = np.where(right_same  == raw_same)
#0: y  1:x
# with open("different.txt",'w'):
#
#
with open("diff.txt",'w') as f:
    for i in range(diff[0].shape[0]):
        f.write(str(int(diff[1][i]))+" "+str(int(diff[0][i]))+"\n")

with open("same.txt",'w') as f:
    for i in range(same[0].shape[0]):
        f.write(str(int(same[1][i]))+" "+str(int(same[0][i]))+"\n")
