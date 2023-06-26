import cv2
import numpy as np

# Load deformation field data
deformation_field = np.load(r'E:\code\python_PK\VoxelMorph-torch-master\Result\temp\0_flow.npy')

# Load image A
# img_C = cv2.imread('reg/flow.tif',0)
img_C = cv2.imread(r'E:\code\python_PK\VoxelMorph-torch-master\Result\temp\0-m.tif',0)

# Get image A dimensions
height, width = img_C.shape

# Create meshgrid for x and y coordinates
x, y = np.meshgrid(np.arange(width), np.arange(height))

# Add deformation field to meshgrid
x_deformed = x + deformation_field[1]
y_deformed = y + deformation_field[0]
# x_deformed = 2 * (x_deformed/(width-1)-0.5)
# y_deformed = 2 * (y_deformed/(height-1)-0.5)
x_deformed = (x_deformed - np.min(x_deformed)) * (4095 / (np.max(x_deformed) - np.min(x_deformed)))
y_deformed = (y_deformed - np.min(y_deformed)) * (2159 / (np.max(y_deformed) - np.min(y_deformed)))

# Remap image A using deformed meshgrid
img_C_deformed = cv2.remap(img_C, x_deformed.astype(np.float32), y_deformed.astype(np.float32), cv2.INTER_LINEAR).astype(np.uint8)

# Display deformed image A
cv2.imwrite(r"E:\code\python_PK\VoxelMorph-torch-master\Result\temp\Deformed_C_norm.tif",img_C_deformed)

deformedC = cv2.imread(r"E:\code\python_PK\VoxelMorph-torch-master\Result\temp\Deformed_C_norm.tif",0)
field_x =- cv2.remap(deformation_field[1],x_deformed.astype(np.float32), y_deformed.astype(np.float32),cv2.INTER_LINEAR )
field_y = - cv2.remap(deformation_field[0],x_deformed.astype(np.float32), y_deformed.astype(np.float32),cv2.INTER_LINEAR )
# x_inv = x + field_x
# y_inv = y + field_y
# x_inv = (x_inv - np.min(x_inv)) * (4095 / (np.max(x_inv) - np.min(x_inv)))
# y_inv = (y_inv - np.min(y_inv)) * (2159 / (np.max(y_inv) - np.min(y_inv)))
x_inv = x - deformation_field[1]
y_inv = y - deformation_field[0]
x_inv = (x_inv - np.min(x_inv)) * (4095 / (np.max(x_inv) - np.min(x_inv)))
y_inv = (y_inv - np.min(y_inv)) * (2159 / (np.max(y_inv) - np.min(y_inv)))
img_C_deformed_inv = cv2.remap(deformedC, x_inv.astype(np.float32), y_inv.astype(np.float32), cv2.INTER_LINEAR).astype(np.uint8)
cv2.imwrite(r"E:\code\python_PK\VoxelMorph-torch-master\Result\temp\Deformed_C_norm_inv_test.tif",img_C_deformed_inv)