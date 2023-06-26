import os
import shutil
import glob

fov = "R003C022"
machine_name = "55.2h_resize_ori"
destDir = rf"E:\data\resize_test\{machine_name}\\"
# os.makedirs(destDir,exist_ok=True)
#destFov = r"\\192.168.7.123\e\NGS\NGS\OutFile\202305231443_Pro23_A_PRM32309260073_SE100_6pMeco_1939\Image\Lane01\*\R001C043_A"
imgs = glob.glob(rf"\\192.168.11.155\e\NGS\NGS\OutFile\202305301717_C2302270013_A_PRM32311080009_PE100_A_HighDensity\Image\Lane01\*\{fov}_A.tif")
for img_path in imgs:
    cyclename = img_path.split("\\")[-2]
    os.makedirs(destDir+"Lane01\\"+cyclename, exist_ok=True)#destDir+"Lane01\\"+cyclename+"\\"+"R001C001_A.tif"
    shutil.copy2(img_path, destDir+"Lane01\\"+cyclename+"\\"+"R001C001_A.tif")
    shutil.copy2(img_path.replace(rf"{fov}_A",rf"{fov}_C"), destDir + "Lane01\\" + cyclename + "\\" + "R001C001_C.tif")
    shutil.copy2(img_path.replace(rf"{fov}_A",rf"{fov}_G"), destDir + "Lane01\\" + cyclename + "\\" + "R001C001_G.tif")
    shutil.copy2(img_path.replace(rf"{fov}_A",rf"{fov}_T"), destDir + "Lane01\\" + cyclename + "\\" + "R001C001_T.tif")