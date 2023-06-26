from torch.utils.tensorboard import SummaryWriter
import torch
import  numpy as np

writer = SummaryWriter(log_dir='callNet_log')




def add_scalars(writer):
     r = 5
     for i in range(100):
         writer.add_scalars(main_tag='scalars1/P1',
                            tag_scalar_dict={'xsinx': i * np.sin(i / r),
                                             'xcosx': i * np.cos(i / r),
                                             'tanx': np.tan(i / r)},
                            global_step=i)
         writer.add_scalars('scalars1/P2',
                            {'xsinx': i * np.sin(i / (2 * r)),
                             'xcosx': i * np.cos(i / (2 * r)),
                             'tanx': np.tan(i / (2 * r))}, i)
         writer.add_scalars(main_tag='scalars2/Q1',
                            tag_scalar_dict={'xsinx': i * np.sin((2 * i) / r),
                                             'xcosx': i * np.cos((2 * i) / r),
                                             'tanx': np.tan((2 * i) / r)},
                            global_step=i)
         writer.add_scalars('scalars2/Q2',
                            {'xsinx': i * np.sin(i / (0.5 * r)),
                             'xcosx': i * np.cos(i / (0.5 * r)),
                             'tanx': np.tan(i / (0.5 * r))}, i)

add_scalars(writer)