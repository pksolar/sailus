import  numpy as np
old_mask = r"E:\code\python_PK\tools\ChangeBasePromoteMapping\mappingTool\fastq\bingji95.09_very_sensitive\deepLearnData\R001C001_mask.npy"
new_mask = r"E:\code\python_PK\tools\ChangeBasePromoteMapping\mappingTool\fastq\bingji95.09_very_sensitive_local\deepLearnData\R001C001_mask.npy"
old = np.load(old_mask)
new = np.load(new_mask)
a = new - old
c = np.where(a == 2)
d = np.where(a == -2)
print("new > old",len(c[0]))
print("new < old",len(d[0]))
print(str(len(d[0]))+" can be corrected")