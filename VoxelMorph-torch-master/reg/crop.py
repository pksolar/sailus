import cv2
import os
import random

# Define the path of the input images
path = r"E:\code\python_PK\VoxelMorph-torch-master\reg\fusionimg\norm"

# Define the output directories
fixed_dir = r"E:\code\python_PK\VoxelMorph-torch-master\reg\fusionimg\norm\fixed_512"
moving_dir = r"E:\code\python_PK\VoxelMorph-torch-master\reg\fusionimg\norm\moving_512"

# Define the size of the cropped images
crop_size = 512

# Define the number of cropped images
num_crops = 200
step = 100
# Define the names of the input images
image_names = ["20_Anorm.tif", "20_Cnorm.tif", "20_Gnorm.tif", "20_Tnorm.tif"]

# Loop through each input image

for i in range(num_crops):
    # Generate a random starting position for the crop
    start_x = random.randint(0, 4096 - crop_size)
    start_y = random.randint(0, 2160 - crop_size)
    # Crop the image
    for image_name in image_names:
        # Read the input image
        image = cv2.imread(os.path.join(path, image_name),0)
        # Get the height and width of the input image
        height, width = image.shape
        # Loop through the number of crops
        crop = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
    # Define the output directory based on the first letter of the image name
        if "_A" in image_name or "_C" in image_name:
            output_dir = fixed_dir
        else:
            output_dir = moving_dir
         # Define the output file name
        output_name = "{}_{}.tif".format(image_name[3], i)
        # Save the cropped image
        cv2.imwrite(os.path.join(output_dir, output_name), crop)