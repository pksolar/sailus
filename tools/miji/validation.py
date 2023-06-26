import numpy as np
fla = np.load("flatten/21_R001C001_label_flatten.npy")
b = np.load("21_R001C001_label.npy")
print(fla[34,3])
print(b[34,3])
if (fla == b).all():
    print("yes")
else:
    print("no")