import cv2
def hist():
    img = cv2.imread("img1_1.tif",0)
    # hist = cv2.calcHist([img], [0], None, [217], [0, 217])
    # 直方图均衡化
    gray_image_eq = cv2.equalizeHist(img,100)
    cv2.imshow("imgout",gray_image_eq)
    cv2.imwrite("imagehjhh.tif",gray_image_eq)
    cv2.waitKey(0)
def adapthist():
    img = cv2.imread("img1_1.tif", 0)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))

    dst = clahe.apply(img) # 将灰度图像和局部直方图相关联

    cv2.imshow("clahe_demo", dst)
    cv2.imwrite("image_adapthist.tif",dst)
    cv2.waitKey(0)
adapthist()