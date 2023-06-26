# python imports
import os
import glob
import warnings
# external imports
import torch
import numpy as np
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from Model import losses
from Model.config import args
from Model.datagenerators import Dataset
from Model.model import U_Network, SpatialTransformer
import cv2

"""
对4张temp图，配准到A temp

"""

def normalize(x):
    Max = np.max(x)
    Min = np.min(x)
    x_ = (x-Min)/(Max-Min)
    return x_


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

def save_flow(flow,name):
    flow_npy = np.squeeze(np.squeeze(flow.cpu().numpy()))
    np.save(name,flow_npy)



def save_image_tensor2cv2(input_tensor: torch.Tensor, cycname,imgname):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    #input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy() #.permute(1, 2, 0).
    # RGB转BRG
    #input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    img = input_tensor.cpu().numpy().astype(np.uint8)
    cycdirname = rf"E:\code\python_PK\VoxelMorph-torch-master\images\testinv"
    if not os.path.exists(cycdirname):
        os.makedirs(cycdirname)
    cv2.imwrite(os.path.join(cycdirname, imgname), img)

    # cv2.imwrite(filename, input_tensor)
    pass


def train():


    #device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')


    # 读入fixed图像
    f_img = normalize(cv2.imread("images/20_round_A.tif",0)[np.newaxis,np.newaxis,:])
    f_img = torch.Tensor(f_img)
    input_fixed = f_img.to(device).float()

    C_img = normalize(cv2.imread("images/20_round_C.tif",0)[np.newaxis,np.newaxis,:])
    G_img = normalize(cv2.imread("images/20_round_G.tif",0)[np.newaxis,np.newaxis,:])
    T_img = normalize(cv2.imread("images/20_round_T.tif", 0)[np.newaxis,np.newaxis, :])
    imglist = [C_img,G_img,T_img]
    # 创建配准网络（UNet）和STN，
    #pk:encoder层是一样的。
    STN = SpatialTransformer((512,512)).to(device)
    STN.eval()

    imgs_path = glob.glob(r"E:\code\python_PK\VoxelMorph-torch-master\reg\phase_imgRound_17_R1C78_resize_ori\Lane01\*\R001C001_A.tif")
    # Get all the names of the training data ,all the path of the training data
    # Training loop.

    #flow load:
    flow0 =torch.Tensor(np.load(r"E:\code\python_PK\VoxelMorph-torch-master\images\testinv\0_50_flow.npy")[np.newaxis,:]).to(device).float()
    with torch.no_grad():


            #imgA =torch.Tensor(normalize(cv2.imread(path,0))[np.newaxis,np.newaxis,:]).to(device).float()
            imgC =torch.Tensor(cv2.imread(r"E:\code\python_PK\VoxelMorph-torch-master\images\testinv\0-50_w.tif",0)[np.newaxis,np.newaxis,:]).to(device).float()

            flow_y = flow0[:,0:1,:,:].clone() #rows
            flow_x = flow0[:,1:,:,:].clone() #cols


            #input_moving = input_moving.to(device).float()

            # img = torch.Tensor(img)
            # input_moving = img.to(device).float()

            # Run the data through the model to produce warp and flow field
            outputfieldy = STN(flow_y,flow0)
            outputfieldx = STN(flow_x, flow0)
            outputfild = torch.cat([flow_y,flow_x],dim=1)
            output0 = STN(imgC, -outputfild)


            # Save model checkpoint
            #save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            #torch.save(UNet.state_dict(), save_file_name)
            # Save images
            # m_name = "Result/temp/"+str(i) + "-round_m.tif"
            # m2f_name = "Result/temp/"+str(i) + "-round_w.tif"
            # f_name = "Result/temp/"+str(i) + "-round_f.tif"
            # flow_name = "Result/temp/"+str(i) + "_round_flow.npy"
            # save_image_tensor2cv2(input_moving,m_name)

            save_image_tensor2cv2(output0, "inv","R001C001_C_outputfild.tif")

            # save_image_tensor2cv2(input_fixed, f_name)

            # save_flow(flow_m2f,flow_name)

            print("warped images have saved.")
            #torch.cuda.empty_cache()



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
