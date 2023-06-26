import cv2
name1 =  "A"
name2 = "C"
name3 =  "G"
name4 = "T"
img1 = cv2.imread(f"img/R001C001_chanel_{name1}.png",0)
img2 = cv2.imread(f"img/R001C001_chanel_{name2}.png",0)
img3 = cv2.imread(f"img/R001C001_chanel_{name3}.png",0)
img4 = cv2.imread(f"img/R001C001_chanel_{name4}.png",0)



_, thresh1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
_, thresh2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
_, thresh3 = cv2.threshold(img3, 127, 255, cv2.THRESH_BINARY)
_, thresh4 = cv2.threshold(img4, 127, 255, cv2.THRESH_BINARY)


result1 = cv2.bitwise_and(thresh1, thresh2)
result2 = cv2.bitwise_and(thresh3, thresh4)
result = cv2.bitwise_and(result1, result2)
number = sum(sum(result/255))
print(sum(sum(result/255)))

cv2.imwrite(f'img/result_all_{number}.png', result)