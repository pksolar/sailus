import numpy as np

import os
import shutil

# ori = np.load(r"E:\data\resize_test\44.1h_resize_ori\res_deep_intent\Lane01\deepLearnData\Cyc069\label\R001C001_label.npy")
# new = np.load(r"E:\data\resize_test\44.1h_resize_ori\intent_only_labelAndmask\Cyc069\label\R001C001_label.npy")
# print("ddd")
# 源文件夹路径
source_folder = r'E:\data\resize_test\08.1h_resize_ori\res_deep_intent\Lane01\deepLearnData'

# 目标文件夹路径
target_folder = r'E:\data\resize_test\08.1h_resize_ori\intent_only_labelAndmask'

os.makedirs(target_folder)
# 遍历源文件夹中的所有子文件夹
for folder_name in os.listdir(source_folder):
    # 构建源文件夹的完整路径
    folder_path = os.path.join(source_folder, folder_name)

    # 检查当前路径是否为文件夹
    if os.path.isdir(folder_path):
        # 构建label文件夹的完整路径
        label_folder_path = os.path.join(folder_path, 'label')

        # 检查label文件夹是否存在
        if os.path.exists(label_folder_path):
            # 构建目标文件夹的完整路径
            target_folder_path = os.path.join(target_folder, folder_name) #

            # 创建文件夹
            os.makedirs(target_folder_path,exist_ok=True)

            #shutil.copytree(folder_path, target_folder_path)

            # 在目标文件夹中创建label文件夹并复制内容
            target_label_folder_path = os.path.join(target_folder_path, 'label')
            shutil.copytree(label_folder_path, target_label_folder_path)