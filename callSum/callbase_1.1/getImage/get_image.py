import os
import glob
import cv2
if __name__ == '__main__':

    i = 20
    j = 0

    for k in range(11):
        if k < 10:
            str_k = '0'+str(k)
        else:
            str_k = str(k)
        file_name1 = r"\\192.168.7.112\e\NGS\Outfile\202211161714_Pro012_ASE30+50_F191FCOLD_ZC_1116RE__\Image\Lane02\Cyc0{}".format(str_k)
        file_name2 = glob.glob(file_name1+"\*.tif")
        for file in file_name2:
            print(file)
            cycle_name = file.split("\\")[-2]
            image_name = file.split("\\")[-1]

            img = cv2.imread(file,0)
            print(os.path.exists(""+cycle_name))
            if not os.path.exists("image_lane02_ori/Lane01/"+cycle_name):
                os.makedirs("image_lane02_ori/Lane01/"+cycle_name)
            save_name = "image_lane02_ori/Lane01/"+cycle_name+"/"+image_name
            cv2.imwrite(save_name,img)
            j = j + 1
            print(j)
            if j > 11:
                j  = 0
                break