import cv2
import numpy as np

# create a white background image
img = np.zeros((480, 480, 3), dtype=np.uint8)
img.fill(255)

# draw grid lines
for i in range(0, 13):
    cv2.line(img, (i * 40, 0), (i * 40, 480), (0, 0, 0), 2)
    cv2.line(img, (0, i * 40), (640, i * 40), (0, 0, 0), 2)

# fill remaining cells with 0
for i in range(12):
    for j in range(12):
        if img[i * 40 + 20, j * 40 + 20, 0] == 255:
            cv2.putText(img, "0", (j * 40 + 10, i * 40 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# fill random cells with values between 0.3 and 0.7
for i in range(10):
    x = np.random.randint(0, 12)
    y = np.random.randint(0, 12)
    value = np.random.uniform(0.3, 0.7)
    cv2.putText(img, f"{value:.1f}", (x * 40 + 10, y * 40 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


cv2.imshow("Grid", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
