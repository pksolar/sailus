import numpy as np

dataset_name = '08'
name_list = ['A','C','G','T']
# E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\Cyc001\label
msk = np.load(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\R001C001_mask.npy".format(dataset_name))
cyc_name = '001'
# msk_flatten = msk.ravel()
value_list = []
sum_right = 0
sum_total = 0
listprint = []
listprint_list=[]
a,b = msk.shape
for i in range(a):
    for j in range(b):
        if msk[i][j] != 0:
            for name in name_list:
                img_norm = np.load(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\Cyc{}\intensity_norm\R001C001_{}.npy".format(dataset_name,cyc_name,name))
                img_norm_v = img_norm[i][j]
                # img = np.load(r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\Cyc052\intensity\R001C001_{}.npy".format(name))
                img_v = np.load(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\Cyc{}\intensity\R001C001_{}.npy".format(dataset_name,cyc_name,name))[i][j]
                msk_v = np.load(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\R001C001_mask.npy".format(dataset_name))[i][j]
                #print("{}:,img_v:{:.2f},  img_norm_v:{:.2f},  msk_v:{}".format(name,img_v,img_norm_v,msk_v))
                value_list.append(img_norm_v)
                listprint=[name,img_v,img_norm_v,msk_v]
                listprint_list.append(listprint)
            label = np.load(r"E:\code\python_PK\callbase\datasets\{}\Res\Lane01\deepLearnData\Cyc{}\label\R001C001_label.npy".format(dataset_name,cyc_name))[i][j]

            #print("pos:",i,",",j,"   label:",label,"  max: ",np.argmax(value_list)+1)
            predict = np.argmax(value_list)+1
            if label == predict:
                sum_right = sum_right+1

            else:
                print("False")
                for i in range(len(listprint_list)):
                    print("{}:,img_v:{:.2f},  img_norm_v:{:.2f},  msk_v:{}".format(*(listprint_list)[i]))
                print("pos:", i, ",", j, "   label:", label, "  max: ", np.argmax(value_list) + 1)
                print("acc:", 100 * sum_right / sum_total)
                print("-------------------------------------------------------------------------------")
            sum_total = sum_total+1
            #print("acc:",100*sum_right/sum_total)
            value_list = []
            listprint_list = []



