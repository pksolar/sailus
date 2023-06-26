import numpy  as np
import glob
import os
machine_name = "17_R1C78_resize_ori"
corrects = glob.glob(rf"E:\data\resize_test\{machine_name}\res_deep_intent\Lane01\deepLearnData\*\label\R001C001_label.npy")
raws = glob.glob(rf"E:\data\resize_test\{machine_name}\res_deep_intent\Lane01\deepLearnData_NoCorrect\*\label\R001C001_label.npy")
# mask = np.load(r"E:\data\resize_test\08_resize_ori\res_deep_intent\Lane01\deepLearnData\R001C001_mask.npy")
i = 0
for correct_path , raw_path in zip(corrects,raws):
    cyclename = correct_path.split("\\")[-3]
    correct = np.load(correct_path)
    raw = np.load(raw_path)
    C = np.equal(correct, raw).astype(int) #不同的位置为0，相同位置为1。和最开始的mask ×一下。
    # C = np.logical_not(C).astype(int)
    os.makedirs(rf"E:\data\resize_test\{machine_name}\res_deep_intent\Lane01\deepLearnData\{cyclename}\msk",exist_ok=True)
    np.save(rf"E:\data\resize_test\{machine_name}\res_deep_intent\Lane01\deepLearnData\{cyclename}\msk\R001C001_msk.npy",C)
    # print("dd")
    i +=1
    print(i)

    # right = np.load(r"E:\data\resize_test\08_resize_ori\res_deep\Lane01\deepLearnData\Cyc010\label\R001C001_label.npy")
    # right_same = right.copy()
    # right_same[right_same == 0] = 10
    # raw = np.load(r"E:\data\resize_test\08_resize_ori\res\Lane01\deepLearnData\Cyc010\label\R001C001_label.npy")
    # raw_same = raw.copy()
    # raw_same[raw_same == 0] = 20
    # diff = np.where(right != raw )
    # same = np.where(right_same == raw_same)