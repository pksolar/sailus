import os
import torch
from datasets import Dataset_3cycle_test
from model.model_NestedUnet import NestedUNet as UNet
from utils import  *
import numpy as np
import json
import fastQ
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
def get_all_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCHSIZE = 1
    test_root_dir = r"C:\deepdata\bigdata\val\44.1h\img"
    fov = test_root_dir.split("\\")[-2]
    # dataset = ImageFolder(train_root_dir)
    # 获取文件夹下所有文件的路径

    test_files = get_all_files(test_root_dir)  # [:100]

    test_dataset = Dataset_3cycle_test(test_files)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False,num_workers=1)

    model = UNet(12).to(device)
    # best_model = torch.load("full_Unet/acc98.8321.pth.tar")['state_dict']
    save_dir = 'pth/'
    model_lists = natsorted(glob.glob(save_dir + '*'))
    model_load = model_lists[-4]
    print(model_load)
    best_model = torch.load(model_load)['state_dict']

    model.load_state_dict(best_model)
    listacgt = []
    labelacgt = []
    dictacgt = {1: "A", 2: "C", 3: "G", 4: "T", 5: "N"}

    accurate_show = AverageMeter()
    loss_val_show = AverageMeter()
    cycle = 1
    size_h = 1360
    size_w = 2560
    with torch.no_grad():
        model.eval()
        for inputs, labels, msk,filename in test_loader:  #注意此处的msk 是否包含mapping上的
            #读图还是还是读大图，切成块以后弄到model里训练。
            model.eval()
            inputs = inputs.to(device)
            labels = labels.to(device)  # labels size;b,c,h,w
            msk = msk.to(device)
            #number = torch.sum(msk)


            #msk = msk.to(device)
            _,_,h,w = inputs.shape

            rows = int( h / size_h)
            cols = int( w / size_w)
            preds_total = None
            for i in range(rows):
                for j in range(cols):
                    print("rows:", i, " cols:", j)
                    input_crop = inputs[:, :, i * size_h:(i + 1) * size_h, j * size_w:(j + 1) * size_w]
                    label_crop = labels[:, :, i * size_h:(i + 1) * size_h, j * size_w:(j + 1) * size_w]
                    msk_crop = msk[:, :, i * size_h:(i + 1) * size_h, j * size_w:(j + 1) * size_w]
                    outputs = model(input_crop)

                    # msk = torch.squeeze(abs(msk))  # msk b,1,b,w

                    _, preds = torch.max(outputs[:, 4:8, :, :],
                                         1)  # preds size [b,h,w],值是0，1,2,3，label是0,1,,,哪个通道的概率值最大，输出是,第几个通道的值最大。，b is batchsize,h,w is the predicted result.save it.
                    _, label_ = torch.max(label_crop[:, 4:8, :, :], 1)  # label的值也是0,1,2,3，
                    # 用mask把不必要的不去比较。mask是，0,1 变成，（-10,0）,然后和label以及preds相加。得到preds和label是负值的地方就是背景，拍平成列表，再删除也元素，得到很短的列表，再去比较。速度会大大加快。
                    # msk_arr =  torch.squeeze(msk_crop).numpy()
                    preds = torch.squeeze(preds).cpu().numpy().flatten().astype(int)
                    mask_crop = torch.squeeze(msk_crop).cpu().numpy().flatten().astype(int)  # msk b,1,b,w
                    label = torch.squeeze(label_).cpu().numpy().flatten().astype(int)

                    idx = np.where(mask_crop != 0)  # 只算mapping上的。
                    preds = preds[
                        idx]  # 去除了背景 .transpose([2, 1, 0])  # numreads,1,channel  .squeeze().transpose() # numreads,channel
                    label = label[idx]
                    try:
                        preds_total = np.concatenate((preds_total, preds), axis=0)
                    except:
                        preds_total = preds

                    try:
                        label_total = np.concatenate((label_total, label), axis=0)
                    except:
                        label_total = label

                    c = (preds == label)
                    accurate = np.sum(c) / label.shape[0] * 100
                    accurate_show.update(accurate)

                    print(" accuray: %.3f %%" % (accurate))
            print("********toatal accurate:{:.3f}%********".format(accurate_show.avg))

            preds_total = preds_total + 1  # 从1开始。
            for i, pred in enumerate(preds_total):
                if cycle == 1:  # 说明是第一个cycle，创建包含几个reads的列表 ，
                    listacgt.append(dictacgt[pred])
                else:  # 第二个cycle以后依次写入：
                    listacgt[i] = listacgt[i] + dictacgt[pred]
            print("cycle:", cycle)
            cycle += 1
            # if cycle >3 :
            #     break
                # print("ergodic use time:",end-start)
        timestr = time.strftime("%m%d%H%M%S")
        dir_name = f'{fov}_{accurate_show.avg:.3f}_{timestr}'
        save_path = f'fastq/{fov}_{accurate_show.avg:.3f}_{timestr}/Lane01//'
        temp_save_path = f'fastq/{fov}_{accurate_show.avg:.3f}_{timestr}/Lane01/sfile//'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(temp_save_path, exist_ok=True)
        fastQ.writeFq(save_path + 'Lane01_fastq.fq', listacgt, 'R001C001')
if __name__ == '__main__':
        main()

