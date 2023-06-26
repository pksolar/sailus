i  = 10
j = 3
string_result = rf"reads:{j},cycle:{i}"
stingddd = "ddddddddddddddddddddddddddd"
with open("changeMapped.txt", "a") as f:
    reads = f.writelines(string_result+"\n")
    reads = f.writelines(stingddd+"\n")