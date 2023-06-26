import shutil
import sys

import numpy as np
"""
use the file in pth to do mapping ,pick the model with highest mapping rate. only the upper half is used for mapping
"""

import os
import torch
from datasets import Dataset_3cycle_test
from model.model_NestedUnet import NestedUNet as UNet
from utils import  *
import numpy as np
import json
import othercode.fastQ as fastQ
import time
from maptool.fastq_Mapping import autoMap
from config import best_mapping_rate
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 定义监测文件夹的路径
folder_path = 'pth'

# 定义需要运行的程序
def run_program(filename):
    # 在这里写入您需要运行的程序的代码
    print("New file detected. Running program...")
    try:
        main(filename)
    except:
        print("bug,wait another")

# 定义事件处理器
class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            # 当有新文件创建时调用run_program()函数
            filename = os.path.basename(event.src_path)
            print(filename + " is get")
            # 调用run_program()函数并传递文件名作为参数
            run_program(filename)

            #run_program()




def get_all_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def main(dir_filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCHSIZE = 1
    test_root_dir = r"E:\data\deepData\test\44.1h\img"
    fov = test_root_dir.split("\\")[-2]
    # dataset = ImageFolder(train_root_dir)
    # 获取文件夹下所有文件的路径

    test_files = get_all_files(test_root_dir)  # [:100]

    test_dataset = Dataset_3cycle_test(test_files)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False,num_workers=0)

    model = UNet(12).to(device)
    # best_model = torch.load("full_Unet/acc98.8321.pth.tar")['state_dict']
    #save_dir = 'pth/'
    #model_lists = natsorted(glob.glob(save_dir + '*'))
    model_load = dir_filename
    end1 = time.time()

    print(model_load)
    best_model = torch.load(model_load)['state_dict']

    model.load_state_dict(best_model)
    listacgt = []
    dictacgt = {1: "A", 2: "C", 3: "G", 4: "T", 5: "N"}

    accurate_show = AverageMeter()
    #loss_val_show = AverageMeter()
    cycle = 1
    size_h = 1360
    size_w = 2560
    with torch.no_grad():
        model.eval()
        for inputs, labels, msk, filename in test_loader:  # 注意此处的msk 是否包含mapping上的
            # 读图还是还是读大图，切成块以后弄到model里训练。
            inputs = inputs.to(device)
            labels = labels.to(device)  # labels size;b,c,h,w
            msk = msk.to(device)
            # number = torch.sum(msk)

            # msk = msk.to(device)
            _, _, h, w = inputs.shape

            rows = int(h / size_h)
            cols = int(w / size_w)
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

                    # print(" accuray: %.3f %%" % (accurate))
            #print("********toatal accurate:{:.3f}%********".format(accurate_show.avg))

            preds_total = preds_total + 1  # 从1开始。
            for i, pred in enumerate(preds_total):
                if cycle == 1:  # 说明是第一个cycle，创建包含几个reads的列表 ，
                    listacgt.append(dictacgt[pred])
                else:  # 第二个cycle以后依次写入：
                    listacgt[i] = listacgt[i] + dictacgt[pred]
            print("cycle:", cycle)
            cycle += 1
            if cycle >30 :
                break
            # print("ergodic use time:",end-start)
        timestr = time.strftime("%m%d%H%M%S")
        dir_name = f'{fov}_{accurate_show.avg:.3f}_{timestr}'
        save_path = f'fastq/{fov}_{accurate_show.avg:.3f}_{timestr}/Lane01//'
        temp_save_path = f'fastq/{fov}_{accurate_show.avg:.3f}_{timestr}/Lane01/sfile//'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(temp_save_path, exist_ok=True)
        fastQ.writeFq(save_path + 'Lane01_fastq.fq', listacgt, 'R001C001')

        mapping_rate = autoMap(save_path)
        #accurate_show

        return  mapping_rate,accurate_show.avg

        #删除产生的mapping文件夹。
        # rootdir = r"E:\code\python_PK\img2base_cnn_seg\fastq"
        # # 验证集会给出machine，fov，acc，time,不必我去读。
        # file_name = dir_name
        # save_root_dir = "C:\deepdata\image_update2"
        # os.makedirs(rootdir + rf"//{file_name}", exist_ok=True)
        # deepUpdateData(rootdir, file_name, save_root_dir)

def automap_delelte(dir_pth_name,idx,epoch):

    #使用pth，运行val_mapping数据集里的图片，得到fastq文件
    mapping_rate,acc = main(dir_pth_name) #生成fastq文件,并且自己map，得到mapping rate
    global best_mapping_rate
    #更新全局变量:
    #best_mapping_rate = max(mapping_rate,best_mapping_rate)
    #将pth_buffer里的pth文件移动到pth文件夹里， 并加上mapping rate
    new_pth_name = "pth/"+"m_"+str(mapping_rate)+rf"_acc{acc:.4f}_idx{idx}_epoch{epoch}.pth.tar"
    print("new_pth_name:",new_pth_name)
    shutil.move(dir_pth_name,new_pth_name)

    #对pth里的文件进行排序，如果超过12个文件，删除一个pth，再进行排序
    save_dir = "pth/"
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > 12:
        os.remove(model_lists[0])  # 删除第一个
        model_lists = natsorted(glob.glob(save_dir + '*'))








if __name__ == '__main__':
    # 创建Observer对象并注册事件处理器
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)

    # 启动监测
    observer.start()

    try:
        while True:
            # 通过sleep来控制监测频率
            time.sleep(1)
    except KeyboardInterrupt:
        # 当按下Ctrl+C时停止监测
        observer.stop()

    # 等待监测线程结束
    observer.join()

