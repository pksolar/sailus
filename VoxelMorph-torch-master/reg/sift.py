import cv2
import numpy as np
# Load the two images
img1 = cv2.imread('img/1.tif',0)
img2 = cv2.imread('img/2.tif',0)
gray1 = img1.copy()
gray2 = img2.copy()

# Convert the images to grayscale
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Find the keypoints and descriptors with SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Match the keypoints
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.9 * n.distance:
        good_matches.append(m)

# Draw the matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the matches
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find the homography matrix
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp the image
h, w = img1.shape[:2]
img2_warped = cv2.warpPerspective(img2, M, (w, h))

# Show the result
cv2.imshow('Result', img2_warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate the translation
tx = M[0, 2]
ty = M[1, 2]

# Print the translation
print('Translation in x direction:', tx)
print('Translation in y direction:', ty)
