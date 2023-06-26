import cv2
import numpy as np
# Create a black image
img = np.zeros((2160,4096), np.uint8)

# Draw horizontal lines
for i in range(0, img.shape[0], 10):
    cv2.line(img, (0, i), (img.shape[1], i), (255), 1)

# Draw vertical lines
for i in range(0, img.shape[1], 10):
    cv2.line(img, (i, 0), (i, img.shape[0]), (255), 1)

# Display the image
cv2.imwrite("flow.tif",img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
