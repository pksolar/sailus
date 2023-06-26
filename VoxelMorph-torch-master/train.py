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


# def save_image(input_moving, f_img,m2f, m_name):
#     path = "Result/base//"
#     input_moving_ = input_moving.permute(0, 2, 3, 1)
#     f_img_ = f_img.permute(0, 2, 3, 1)
#     m2f_ = m2f.permute(0, 2, 3, 1)
#     # input_moving_img = input_moving.cpu().numpy()[]
#     for i in range(10):
#         res = k[i]  # 得到batch中其中一步的图片
#         image = Image.fromarray(np.uint8(res)).convert('RGB')
#         # image.show()
#         # 通过时间命名存储结果
#         timestamp = datetime.datetime.now().strftime("%M-%S")
#         savepath = timestamp + '_r.jpg'
#         image.save(savepath)


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
    f_img = cv2.imread("images/0001.jpg")[np.newaxis,:]
    #这个input_fixed将f_img又包裹了两层。 省略号代表原来的它。
    #input_fixed =
    #
    vol_size = 2 #vol_size 取的是input_fixed 's last  deimensions
    # [B, C, D, W, H] batch，channel，Depth，width，height  宽高深   ，channel 是1
    #input_fixed = np.repeat(input_fixed, args.batch_size, axis=0) #在0维进行扩充，扩充行。0维是最高维。
    #input_fixed = torch.from_numpy(input_fixed).to(device).float()#最后将input_fixed变成tensor,



    # 创建配准网络（UNet）和STN，
    #pk:encoder层是一样的。
    nf_enc = [16,32, 32, 32, 32]
    #pk:以下下是decoder不同层。
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else: #否则就是vm2
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    UNet = U_Network(2, nf_enc, nf_dec).to(device) # len(vol_size )是数据维度。
    STN = SpatialTransformer((512,512)).to(device)

    #进入训练模式：
    UNet.train()
    STN.train()
    # 模型参数个数
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))


    # Set optimizer and losses
    opt = Adam(UNet.parameters(), lr=args.lr)

    #pk:自己定义loss函数：这就是关键，loss 由两个部分组成： 第二个为了平滑。
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    #sim_loss_fn = losses.mse_loss
    grad_loss_fn = losses.gradient_loss2d

    # Get all the names of the training data ,all the path of the training data
    train_files_fixed = glob.glob(os.path.join(args.train_dir_fixed, '*tif'))
    train_files_moving = glob.glob(os.path.join(args.train_dir_moving, '*.tif'))
    DS = Dataset(files_fixed=train_files_fixed,files_moving=train_files_moving)
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Training loop.
    for i in range(1, args.n_iter + 1):
        # Generate the moving images and convert them to tensors.
        input_fixed,input_moving = iter(DL).next()
        # [B, C, D, W, H]
        input_moving = input_moving.to(device).float()
        input_fixed = input_fixed.to(device).float()

        # Run the data through the model to produce warp and flow field
        flow_m2f,_ = UNet(input_moving, input_fixed)
        m2f = STN(input_moving, flow_m2f)

        # Calculate loss
        sim_loss = sim_loss_fn(m2f, input_fixed)
        grad_loss = grad_loss_fn(flow_m2f)
        loss = sim_loss + args.alpha * grad_loss


        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % args.n_save_iter == 0:
            # Save model checkpoint
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(UNet.state_dict(), save_file_name)
            # Save images
            m_name = "Result/base/"+str(i) + "-m.tif"
            m2f_name = "Result/base/"+str(i) + "-w.tif"
            f_name = "Result/base/"+str(i) + "-f.tif"
            save_image_tensor2cv2(input_moving,m_name)
            save_image_tensor2cv2(m2f,m2f_name)
            save_image_tensor2cv2(input_fixed, f_name)
            print("i: %d  loss: %f  sim: %f  grad: %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()),
                  flush=True)
            print("%d, %f, %f, %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), file=f)
            print("warped images have saved.")
    f.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
