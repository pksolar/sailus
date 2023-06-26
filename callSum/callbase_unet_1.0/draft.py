# def load_image_dataset(dataset_dir):
# """
# 加载图片数据集
# :param dataset_dir: 数据集所在目录
# :return: 图片数据集
# """
# # 获取数据集目录下的所有文件
# files = os.listdir(dataset_dir)
# # 将文件路径拼接成完整的文件路径
# file_paths = [os.path.join(dataset_dir, file) for file in files]
# # 使用 OpenCV 读取图片
# images = [cv2.imread(file_path,0) for file_path in file_paths]
#
#
# return dataset