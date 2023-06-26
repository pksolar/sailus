import shutil
for i in range(1,21):
    shutil.copy2("changeAndMapping.py",fr"programDir/changeAndMapping_{i}.py")
for i in range(1,21):
    with open(rf"programDir/changeAndMapping_{i}.py","r") as f:
        abs = f.readlines()
        abs[]