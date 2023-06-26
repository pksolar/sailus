from __future__ import division

import glob
import os
import warnings

from Networks.HR_Net.seg_hrnet import get_seg_model

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import dataset
import math
from image import *
from utils import *

import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
import time
logger = logging.getLogger('mnist_AutoML')
setup_seed(args.seed)
test_list = glob.glob(r"E:\code\python_PK\tools\hrnet_label\val\imgdata_full\*.npy")
tuner_params = nni.get_next_parameter()
logger.debug(tuner_params)
params = vars(merge_parameter(return_args, tuner_params))

def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        # print(fname)
        img = load_data_fidt_test(Img_path, args, train)

        blob = {}
        blob['img'] = img

        blob['gt'] = None
        blob['fname'] = fname
        blob['msk'] = None
        data_keys[count] = blob
        count += 1

    return data_keys
def show_map(input,name):
    dict_acgt = {0:"A",1:"C",2:"G",3:"T"}
    input_npy = input.squeeze().cpu().numpy()
    for i in range(input_npy.shape[0]):
        input_ = input_npy[i,:,:]
        input_[input_ < 0] = 0
        # input_ = input_[0][0]
        fidt_map1 = input_
        fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
        fidt_map1 = fidt_map1.astype(np.uint8)+1
        #fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
        os.makedirs(f"result/{name}",exist_ok=True)
        cv2.imwrite(f"result/{name}/R001C001_{dict_acgt[i]}.tif",fidt_map1)


def validate(Pre_data, model, args):
    print('begin test')
    mse_show = AverageMeter()
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data,
                            shuffle=False,
                            args=args, train=False),
        batch_size=1)
    model.eval()
    criterion = torch.nn.MSELoss().cuda()

    if not os.path.exists('./local_eval/loc_file'):
        os.makedirs('./local_eval/loc_file')

    for i, (fname, img) in enumerate(test_loader):

        count = 0
        img = img.cuda()



        with torch.no_grad():
            d6 = model(img)
            #mse = criterion(d6*msk,gt*msk)
            #print(f"val_mse:{mse.item()}")
        #mse_show.update(mse.item())
        name = fname[0].replace(".npy","")
        show_map(d6,name)
        #show_map(img,name+"_img")
        # show_map(gt,name+"_gt")
    return mse_show.avg

model = get_seg_model(train=True)
model = nn.DataParallel(model, device_ids=[0])
model = model.cuda()
best_model = torch.load("checkpoint/mse0.00594.pth")['state_dict']
model.load_state_dict(best_model)
test_data = pre_data(test_list, args, train=False)
mse = validate(test_data, model, params)