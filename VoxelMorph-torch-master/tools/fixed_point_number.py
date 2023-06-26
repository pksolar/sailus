import cv2

name_list = ["A","C","G","T"]
for name1 in name_list:
    for name2 in name_list:
        img1 = cv2.imread(f"img/R001C001_chanel_{name1}.png",0)
        if name2 == name1:
            break
        img2 = cv2.imread(f"img/R001C001_chanel_{name2}.png",0)



        _, thresh1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

        result = cv2.bitwise_and(thresh1, thresh2)
        number = sum(sum(result/255))
        print(sum(sum(result/255)))

        cv2.imwrite(f'img/result_{name1}_{name2}_{number}.png', result)