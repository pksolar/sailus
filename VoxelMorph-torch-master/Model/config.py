import argparse

parser = argparse.ArgumentParser()

# 公共参数
parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0')


parser.add_argument("--atlas_file", type=str, help="gpu id number",
                    dest="atlas_file", default=r'G:\data\tjall\06quchu\cbct\0.nii.gz')
parser.add_argument("--model", type=str, help="voxelmorph 1 or 2",
                    dest="model", choices=['vm1', 'vm2'], default='vm2')
parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default='Result')

# train时参数
#data is in: G:\data\tjall\06quchu
parser.add_argument("--train_dir_fixed", type=str, help="data folder with training vols",
                    dest="train_dir_fixed", default=r"reg/fixed1024")
parser.add_argument("--train_dir_moving", type=str, help="data folder with training vols",
                    dest="train_dir_moving", default=r"reg/moving1024")
parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=4e-4)
parser.add_argument("--n_iter", type=int, help="number of iterations",
                    dest="n_iter", default=150000)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc",
                    dest="sim_loss", default='ncc')
parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=5)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
parser.add_argument("--n_save_iter", type=int, help="frequency of model saves",
                    dest="n_save_iter", default=1000)
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default='Checkpoint')
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default='Log')

# test时参数
parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='images/moving_round_512')
parser.add_argument("--label_dir", type=str, help="label data directory",
                    dest="label_dir", default='images/fixed_round_512')
parser.add_argument("--checkpoint_path", type=str, help="model weight file",
                    dest="checkpoint_path", default="Checkpoint/434.pth")

args = parser.parse_args()