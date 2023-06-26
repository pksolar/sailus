import os
import glob
import cv2


# ori4： \\192.168.7.108\e\NGS\OutFile\202211081816_Pro008_A_SE100_5pM_originBuffer_75-42CV1.24.14_SPM22210270053__\Image\Lane01\Cyc001

if __name__ == '__main__':

    i = 20
    j = 0

    for k in range(100):
        if k < 10:
            str_k = '0'+str(k)
        else:
            str_k = str(k)
        file_name1 = r"\\192.168.7.108\e\NGS\OutFile\202211091509_Pro008_B_SE100_3pM_4xcore_RPA+Aftertreat_ecoli_CH1028-X3__\Image\Lane01\Cyc0{}".format(str_k)
        file_name2 = glob.glob(file_name1+"\*.tif")
        for file in file_name2:
            print(file)
            cycle_name = file.split("\\")[-2]
            image_name = file.split("\\")[-1]

            img = cv2.imread(file,0)
            print(os.path.exists("image_ori5/Lane01/"+cycle_name))
            if not os.path.exists("image_ori5/Lane01/"+cycle_name):
                os.makedirs("image_ori5/Lane01/"+cycle_name)
            save_name = "image_ori5/Lane01/"+cycle_name+"/"+image_name
            cv2.imwrite(save_name,img)
            # j = j + 1
            # print(j)
            # if j > 11:
            #     j  = 0
            #     break