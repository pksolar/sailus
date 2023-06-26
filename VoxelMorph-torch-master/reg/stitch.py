import cv2
import os
import numpy as np

def stitch_images(input_dir, output_image_path, width, height):
    # 创建一个空白图像用于存放拼接后的大图
    stitched_image = np.zeros((height, width), dtype=np.uint8)

    # 遍历输入文件夹中的所有子图
    for filename in os.listdir(input_dir):
        # 从文件名中获取子图的左上角坐标
        if filename.startswith("C"):
            if filename.endswith(".tif"):
                _,x, y,_ = filename[:-4].split('_')
                x = int(x)
                y =  int(y)

                # 读取子图
                sub_image = cv2.imread(os.path.join(input_dir, filename),0)

                # 将子图粘贴到大图上
                stitched_image[y:y + 1024, x:x + 1024] = sub_image

    # 保存拼接后的大图
    cv2.imwrite(output_image_path, stitched_image)

# 输入文件夹
input_dir = r"E:\code\python_PK\VoxelMorph-torch-master\Result\temp\bspl_crop"

# 输出图像路径
output_image_path = r"E:\code\python_PK\VoxelMorph-torch-master\Result\temp\bspl_C.tif"

# 原始大图的宽度和高度
width = 4096
height = 2160

# 拼接子图并保存为大图
stitch_images(input_dir, output_image_path, width, height)
