import cv2
import numpy as np
base = "T"
# Load deformation field data
deformation_field = np.load(rf'flow_grid_50.npy')

# Load image A
# img_C = cv2.imread('reg/flow.tif',0)
img_C = cv2.imread(rf'R001C001_A.tif',0)

# Get image A dimensions
height, width = img_C.shape

# Create meshgrid for x and y coordinates
x, y = np.meshgrid(np.arange(width), np.arange(height))

# Add deformation field to meshgrid
x_deformed = x + deformation_field[0]
y_deformed = y + deformation_field[1]
# x_deformed = 2 * (x_deformed/(width-1)-0.5)
# y_deformed = 2 * (y_deformed/(height-1)-0.5)
# x_deformed = (x_deformed - np.min(x_deformed)) * (4096 / (np.max(x_deformed) - np.min(x_deformed)))
# y_deformed = (y_deformed - np.min(y_deformed)) * (2160 / (np.max(y_deformed) - np.min(y_deformed)))

# Remap image A using deformed meshgrid
img_C_deformed = cv2.remap(img_C, x_deformed.astype(np.float32), y_deformed.astype(np.float32), cv2.INTER_LINEAR).astype(np.uint8)

# Display deformed image A
# cv2.imshow('Deformed_bspl_ C', img_C_deformed)
cv2.imwrite(rf"R001C001_A_flow_use_grid_50_norm.tif",img_C_deformed)
