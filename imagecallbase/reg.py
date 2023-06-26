import  cv2
img1 = cv2.imread("1.tif",0)
img2 = cv2.imread("2.tif",0)
# t1,a1 = cv2.threshold(img1,)
ret, otsu1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret, otsu2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite("1-1.tif",otsu1)
cv2.imwrite("2-1.tif",otsu2)

cv2.imshow("img1",otsu1)
cv2.imshow("img2",otsu2)
cv2.waitKey(0)