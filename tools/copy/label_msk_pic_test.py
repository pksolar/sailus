import numpy as np
x= 4
y = 2497

name_list = ['A','C','G','T']
msk = np.load(r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\R001C001_mask.npy")
msk_flatten = msk.ravel()
value_list = []
sum_right = 0
sum_total = 0
for idx,i in enumerate(msk_flatten):
 if i == 1 :
    for name in name_list:
        img_norm = np.load(r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\Cyc001\intensity_norm\R001C001_{}.npy".format(name))
        img_norm_v = img_norm.ravel()[idx]
        # img = np.load(r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\Cyc052\intensity\R001C001_{}.npy".format(name))
        img_v = np.load(r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\Cyc001\intensity\R001C001_{}.npy".format(name)).ravel()[idx]

        msk_v = np.load(r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\R001C001_mask.npy").ravel()[idx]

        print("{}:,img_v:{:.2f},  img_norm_v:{:.2f},  msk_v:{}".format(name,img_v,img_norm_v,msk_v))
        value_list.append(img_norm_v)
    label = np.load(r"E:\code\python_PK\callbase\datasets\08\Res\Lane01\deepLearnData\Cyc001\label\R001C001_label.npy").ravel()[idx]
    print("label:",label)
    predict = np.argmax(value_list)+1
    print("max:",np.argmax(value_list)+1)
    if label == predict:
        sum_right = sum_right+1
    sum_total = sum_total+1
    print("acc:",100*sum_right/sum_total)
    value_list = []



