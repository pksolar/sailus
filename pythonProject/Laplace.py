import cv2
import glob
import os
if __name__ == '__main__':

    i = 3
    j = 0

    for k in range(11):
        if k < 10:
            str_k = '0'+str(k)
        else:
            str_k = str(k)
        file_name1 = r"image_ori\Lane01\Cyc0{}".format(str_k)
        file_name2 = glob.glob(file_name1+"\*.tif")
        for file in file_name2:
            #print(file)
            cycle_name = file.split("\\")[-2]
            image_name = file.split("\\")[-1]
            blockSize = 256

            img = cv2.imread(file,0)

            #dst = unevenLightCompensate(img, blockSize)
            dst = cv2.Laplacian(img,cv2.CV_64F)
            cv2.imshow("lap",dst)
            cv2.waitKey(0)
            #result = np.concatenate([img, dst], axis=1)
            name = "lap"
            # cv2.imshow('image/result'.format(ele), result)
            print(os.path.exists("image_{}/Lane01/".format(name)+cycle_name))
            if not os.path.exists("image_{}/Lane01/".format(name)+cycle_name):
                os.mkdir("image_{}/Lane01/".format(name)+cycle_name)
            save_name = "image_{}/Lane01/".format(name)+cycle_name+"/"+image_name
            cv2.imwrite(save_name,dst)
            j = j + 1
            print(j)
            if j > 11:
                j  = 0
                break
