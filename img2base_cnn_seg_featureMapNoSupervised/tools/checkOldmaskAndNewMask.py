import  numpy as np
old_mask = r"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\mappingToolTest_total\change_sensitive_local\Lane01\deepLearnData\R001C001_mask.npy"
new_mask = r"E:\code\python_PK\img2base_cnn_seg_featureMapNoSupervised\fastq\mappingToolTest_total\normal\Lane01\deepLearnData\R001C001_mask.npy"
old = np.load(old_mask)
new = np.load(new_mask)
a = new - old
c = np.where(a == 2)
d = np.where(a == -2)
print("hello world")