import cv2
import os
import random





def crop_and_save_image_blocks(img_path, output_folder,image_name, block_size=(512, 512), step_size=(100, 100)):
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # 获取图像尺寸
    img_height, img_width = img.shape[:2]

    # 按照给定的步长裁剪图像
    for y in range(0, img_height - block_size[1] + 1, step_size[1]):
        for x in range(0, img_width - block_size[0] + 1, step_size[0]):
            block = img[y:y + block_size[1], x:x + block_size[0]]
            output_path = os.path.join(output_folder, f"{image_name}_{x}_{y}.tif")
            cv2.imwrite(output_path, block)

if __name__ == "__main__":
    # Define the path of the input images
    path = r"E:\code\python_PK\VoxelMorph-torch-master\images\\"

    # Define the output directories
    fixed_dir = r"E:\code\python_PK\VoxelMorph-torch-master\reg\fusionimg\norm\fixed_512"
    moving_dir = r"E:\code\python_PK\VoxelMorph-torch-master\reg\fusionimg\norm\moving_512"

    # Define the names of the input images
    image_names = ["20_round_A", "20_round_C", "20_round_G", "20_round_T"]




    for image_name in image_names:
        img_path = path + f"{image_name}.tif"
        output_folder_fixed = path+"fixed_round_512"
        output_folder_moving = path + "moving_round_512"

        if "_A" in img_path:
            output_folder = output_folder_fixed
        else:
            output_folder = output_folder_moving

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


        crop_and_save_image_blocks(img_path, output_folder,image_name[-1])
