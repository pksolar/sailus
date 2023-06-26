import os
import shutil
import glob
source_fov = "R004C022"
sources = glob.glob(rf"\\192.168.11.152\e\NGS\NGS\OutFile\202306151759_C522302270010_B_PRM32311150042_SE100_HighDensity\Image\Lane01\*\{source_fov}_A.tif")

save_dir_name = "52.2h_resize_ori"
destDir = rf"E:\data\resize_test\{save_dir_name}\\"
# os.makedirs(destDir,exist_ok=True)
#destFov = r"\\192.168.7.123\e\NGS\NGS\OutFile\202305231443_Pro23_A_PRM32309260073_SE100_6pMeco_1939\Image\Lane01\*\R001C043_A"

for img_path in sources:
    cyclename = img_path.split("\\")[-2]
    print(cyclename)
    if cyclename == "Cyc101":
        break

    # if int(cyclename[-3:])<69:
    #     continue
    os.makedirs(destDir+"Lane01\\"+cyclename, exist_ok=True)#destDir+"Lane01\\"+cyclename+"\\"+"R001C001_A.tif"
    shutil.copy2(img_path, destDir+"Lane01\\"+cyclename+"\\"+"R001C001_A.tif")
    shutil.copy2(img_path.replace(rf"{source_fov}_A",rf"{source_fov}_C"), destDir + "Lane01\\" + cyclename + "\\" + "R001C001_C.tif")
    shutil.copy2(img_path.replace(rf"{source_fov}_A",rf"{source_fov}_G"), destDir + "Lane01\\" + cyclename + "\\" + "R001C001_G.tif")
    shutil.copy2(img_path.replace(rf"{source_fov}_A",rf"{source_fov}_T"), destDir + "Lane01\\" + cyclename + "\\" + "R001C001_T.tif")