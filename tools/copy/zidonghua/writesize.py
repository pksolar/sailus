import numpy as np
with open("E:\software\sailu\commen\config.ini") as f:
   res = f.readlines()
   res[6] = "imWidth=999\n"
   res[7] = "imHeight=222"

with open("E:\software\sailu\commen\config.ini",'w') as fw:
   for line in res:
      fw.write(line)

