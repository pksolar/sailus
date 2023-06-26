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



def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
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
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy() #.permute(1, 2, 0).
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)
    pass


def train():
    # 创建需要的文件夹并指定gpu
    make_dirs()
    #device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    # 日志文件
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    # 读入fixed图像
    f_img = normalize(cv2.imread("images/20_Anorm.tif",0)[np.newaxis,np.newaxis,:])
    f_img = torch.Tensor(f_img)
    input_fixed = f_img.to(device).float()

    C_img = normalize(cv2.imread("images/20_Cnorm.tif",0)[np.newaxis,np.newaxis,:])
    G_img = normalize(cv2.imread("images/20_Gnorm.tif",0)[np.newaxis,np.newaxis,:])
    T_img = normalize(cv2.imread("images/20_Tnorm.tif", 0)[np.newaxis,np.newaxis, :])
    imglist = [C_img,G_img,T_img]
    # 创建配准网络（UNet）和STN，
    #pk:encoder层是一样的。
    nf_enc = [16,32, 32, 32, 32]
    #pk:以下下是decoder不同层。
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else: #否则就是vm2
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    UNet = U_Network(2, nf_enc, nf_dec).to(device) # len(vol_size )是数据维度。
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    UNet.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    STN = SpatialTransformer((2160,4096)).to(device)


    UNet.eval()
    STN.eval()


    #进入训练模式：
    UNet.train()
    STN.train()
    # 模型参数个数
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))


    # Get all the names of the training data ,all the path of the training data
    # Training loop.
    with torch.no_grad():
        for i,img  in enumerate(imglist):
            # Generate the moving images and convert them to tensors.

            #input_moving = input_moving.to(device).float()

            img = torch.Tensor(img)
            input_moving = img.to(device).float()

            # Run the data through the model to produce warp and flow field
            flow_m2f,_ = UNet(input_moving, input_fixed)
            m2f = STN(input_moving, flow_m2f)
            # Save model checkpoint
            #save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            #torch.save(UNet.state_dict(), save_file_name)
            # Save images
            m_name = "Result/temp/"+str(i) + "-m.tif"
            m2f_name = "Result/temp/"+str(i) + "-w.tif"
            f_name = "Result/temp/"+str(i) + "-f.tif"
            flow_name = "Result/temp/"+str(i) + "_flow.npy"
            save_image_tensor2cv2(input_moving,m_name)
            save_image_tensor2cv2(m2f,m2f_name)
            save_image_tensor2cv2(input_fixed, f_name)

            save_flow(flow_m2f,flow_name)

            print("warped images have saved.")
            #torch.cuda.empty_cache()
        f.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
