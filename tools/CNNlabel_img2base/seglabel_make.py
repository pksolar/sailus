import numpy as np
#读入label
import glob
from scipy.ndimage import grey_dilation, generate_binary_structure

path_labels = glob.glob(r"E:\code\python_PK\callbase\datasets\21\Res\Lane01\deepLearnData\*\label\R001C001_label.npy")

for path_label in path_labels:
    label = np.load(path_label)
    label[label == 5] = 0
    struct = generate_binary_structure(2, 2)
    ones_matrix = grey_dilation(label, footprint=struct)
    print("hello")