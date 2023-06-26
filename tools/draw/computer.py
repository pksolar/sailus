import cv2
import numpy as np

# 创建一个黑色背景图像
img = np.zeros((400, 400, 3), dtype=np.uint8)

# 绘制篮球
orange_color = (0, 127, 255)
center = (int(img.shape[1]/2), int(img.shape[0]/2))
radius = 100
cv2.circle(img, center, radius, orange_color, -1)
cv2.circle(img, center, radius, (0, 0, 0), 3)

# 绘制篮球上的线条
line_color = (255, 255, 255)
line_thickness = 3
cv2.line(img, (center[0]-radius, center[1]), (center[0]+radius, center[1]), line_color, line_thickness)
cv2.line(img, (center[0], center[1]-radius), (center[0], center[1]+radius), line_color, line_thickness)
cv2.ellipse(img, center, (radius, radius), 0, 0, 180, line_color, line_thickness)
cv2.ellipse(img, center, (radius, radius), 0, 180, 360, line_color, line_thickness)

# 显示结果
cv2.imshow('Basketball', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
