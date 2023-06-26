import cv2
import os


def crop_and_save(input_image_path, output_dir_fixed, output_dir_moving):
    # 读取输入图像
    image = cv2.imread(input_image_path,0)

    # 计算每份子图的宽度和高度的步长
    width_step = (image.shape[1] - 1024) // 4
    height_step = (image.shape[0] - 1024) // 2

    # 选择输出目录
    output_dir = output_dir_fixed if "_A" in input_image_path else output_dir_moving

    # 裁剪子图并保存
    name =input_image_path.split(".")[0][-1]
    for i in range(3):
        for j in range(5):
            x = j * width_step
            y = i * height_step

            # 处理图像边缘，确保裁剪区域不超出原始图像范围
            x = min(x, image.shape[1] - 1024)
            y = min(y, image.shape[0] - 1024)

            cropped_image = image[y:y + 1024, x:x + 1024]
            output_path = os.path.join(output_dir, f"{name}_{x}_{y}.tif")
            cv2.imwrite(output_path, cropped_image)


if __name__ == "__main__":
    # Define the path of the input images
    path = r"E:\code\python_PK\VoxelMorph-torch-master\images\\"

    # Define the names of the input images
    image_names = ["20_round_A", "20_round_C", "20_round_G", "20_round_T"]




    for image_name in image_names:
        img_path = path + f"{image_name}.tif"
        output_dir_fixed = "fixed1024"
        output_dir_moving = "moving1024"

        # 如果文件夹不存在，则创建
        os.makedirs(output_dir_fixed, exist_ok=True)
        os.makedirs(output_dir_moving, exist_ok=True)

        if "_A" in img_path:
            output_folder = output_dir_fixed
        else:
            output_folder = output_dir_moving

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


        crop_and_save(img_path, output_dir_fixed, output_dir_moving)