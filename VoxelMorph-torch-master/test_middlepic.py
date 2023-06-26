# python imports
import os
import glob
# external imports
import torch
import numpy as np
#import torchsnooper
import SimpleITK as sitk
# internal imports
from Model import losses
from Model.config import args
from Model.model import U_Network, SpatialTransformer
from Model.datagenerators import Dataset
import torch.utils.data as Data
from PIL import  Image

import matplotlib.pyplot as plt

def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))
def save_image_forf(img, ref_img, name):
    img = sitk.GetImageFromArray(img.cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))

def visualize(x_enc,ref_img):
    #每个通道取中间张
    j = 0
    for x in x_enc: #B,C,D,W,H
        B,C,D,W,H = x.shape
        j = j+1
        for i in range(C):#每个通道的中间帧
            print("i",i)
            pic = x[0,i,round(D/2),:,:].detach().numpy() #[W,H]
            plt.imshow(pic,cmap='gray')
            plt.imsave("Result/pic5cbct/%d_%d.png"%(j,i),pic,cmap='gray')
            #转图像:numpy,0-1  --> 255 ->plt
        #直接保存为nii
            # name = "pic5ct/%d_%d.nii.gz"%(j,i)
            # x_nii = x[0,i,:,:,:]
            # save_image_forf(x_nii,ref_img,name)

def make_dirs():
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)


# @torchsnooper.snoop()
def test():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(args.checkpoint_path)

    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    # set up atlas tensor
    input_fixed = torch.from_numpy(input_fixed).to(device).float()

    # Test file and anatomical labels we want to evaluate
    test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))
    print("The number of test data: ", len(test_file_lst))

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16,32, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    # Set up model
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    UNet.load_state_dict(torch.load(args.checkpoint_path,map_location='cpu'))
    STN_img = SpatialTransformer(vol_size).to(device)

    UNet.eval()
    STN_img.eval()


    DSC = []
    # fixed图像对应的label


    train_files_fixed = glob.glob(os.path.join(args.train_dir_fixed, '*.nii.gz'))
    train_files_moving = glob.glob(os.path.join(args.train_dir_moving, '*.nii.gz'))
    DS = Dataset(files_fixed=train_files_fixed, files_moving=train_files_moving)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    ref_img = sitk.ReadImage(r'G:\data\tjall\06quchu\cbct\156.nii.gz')

    img_arr_fixed = sitk.GetArrayFromImage(sitk.ReadImage(r'G:\data\tjall\06quchu\cbct\156.nii.gz'))[np.newaxis,np.newaxis, ...]
    img_fixed = torch.from_numpy(img_arr_fixed).to(device).float()

    # # [B, C, D, W, H]
    # input_moving = input_moving.to(device).float()
    # input_fixed = input_fixed.to(device).float()

    flow_m2f ,x_enc= UNet(img_fixed, img_fixed)
    visualize(x_enc,ref_img)
    print("hello world")



if __name__ == "__main__":
    test()
