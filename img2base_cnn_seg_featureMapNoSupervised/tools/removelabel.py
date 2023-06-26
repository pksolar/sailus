import os


def remove_label_from_filenames(folder_path):
    # 获取指定文件夹下的所有文件名
    filenames = os.listdir(folder_path)

    for filename in filenames:
        if 'label' in filename:
            new_filename = filename.replace('_label', '')
            # 构建原文件的完整路径
            old_filepath = os.path.join(folder_path, filename)
            # 构建新文件的完整路径
            new_filepath = os.path.join(folder_path, new_filename)
            # 重命名文件
            os.rename(old_filepath, new_filepath)
            print(f"Renamed {filename} to {new_filename}")


# 指定文件夹路径
folder_path = r'E:\data\testAuto\label\\'
# 调用函数去除文件名中的"label"字符
remove_label_from_filenames(folder_path)