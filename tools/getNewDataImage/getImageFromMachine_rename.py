import os
import shutil
import glob

fov = "R002C034"
machine_name = "08.2h_resize_ori"
destDir = rf"E:\data\resize_test\{machine_name}\\"
# os.makedirs(destDir,exist_ok=True)
#destFov = r"\\192.168.7.123\e\NGS\NGS\OutFile\202305231443_Pro23_A_PRM32309260073_SE100_6pMeco_1939\Image\Lane01\*\R001C043_A"
imgs = glob.glob(rf"E:\code\python_PK\callbase\datasets\highDens_08\Image\Lane01\*\{fov}_A.tif")
for img_path in imgs:
    cyclename = img_path.split("\\")[-2]
    os.makedirs(destDir+"Lane01\\"+cyclename, exist_ok=True)#destDir+"Lane01\\"+cyclename+"\\"+"R001C001_A.tif"
    shutil.copy2(img_path, destDir+"Lane01\\"+cyclename+"\\"+"R001C001_A.tif")
    shutil.copy2(img_path.replace(rf"{fov}_A",rf"{fov}_C"), destDir + "Lane01\\" + cyclename + "\\" + "R001C001_C.tif")
    shutil.copy2(img_path.replace(rf"{fov}_A",rf"{fov}_G"), destDir + "Lane01\\" + cyclename + "\\" + "R001C001_G.tif")
    shutil.copy2(img_path.replace(rf"{fov}_A",rf"{fov}_T"), destDir + "Lane01\\" + cyclename + "\\" + "R001C001_T.tif")