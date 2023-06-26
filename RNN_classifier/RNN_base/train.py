import numpy as np
import torch
import glob
from datasets_rnn import Dataset_epoch,Dataset_npy,Dataset_npy2
import torch.utils.data as data
import time
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        #print("self.avg:",self.avg)

loss_show = AverageMeter()
accurate_show = AverageMeter()
batchsize  = 30000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class Model(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
#         super(Model, self).__init__()
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.rnn = torch.nn.RNN(input_size=self.input_size,
#                                 hidden_size=self.hidden_size,
#                                 num_layers=num_layers,batch_first = True)
#
#     def forward(self, input):
#         hidden = torch.zeros(self.num_layers,
#                              self.batch_size,
#                              self.hidden_size).to(device)
#         out, _ = self.rnn(input, hidden) #out shape: 4,3,4
#         return out

class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True): #hidden size 100
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.gru = torch.nn.LSTM(input_size, hidden_size, n_layers,bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size) #双向 所以乘
    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,batch_size, self.hidden_size).to(device)
        return hidden

    def forward(self, input, seq_lengths=3):
        # input shape : B x S -> S x B
        #input = input.t()
        input = input.permute(1,0,2).contiguous()
        batch_size = batchsize
        hidden = self._init_hidden(batch_size)
        # pack them up

        output, hidden = self.gru(input,( hidden,hidden))
        hidden = hidden[0]
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output

def main():

    dataset_name = '08'

    dictacgt = {1:'A',2:"C",3:"G",4:"T"}
    rate = 1
    """
    输入的数据is npy : num x cycle x channel     25w x 100 x 4
    seq_len is 3 
    two direction
    
    
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_array = np.load("../08_R001C001_pure.npy")[:224496]
    label_array = np.load("../08_R001C001_label_pure.npy")[:224496]
    seq_len = 3
    train_dataset = Dataset_npy(data_array,label_array,seq_len)

    train_loader = data.DataLoader(train_dataset,batch_size=batchsize,num_workers=0,shuffle=False,drop_last = True)

    net = RNNClassifier(4,100,4, 2,True).to(device) #def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
    for epoch in range(10000):
        for inputs,labels in train_loader: #b,seq,features
            inputs = inputs.to(device) #b,3,4
            labels = labels[:,1,:].to(device) #b,4

            outputs = net(inputs)#b,4
            #outputs = inputs[:,1,:]
            loss = criterion(outputs, labels)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # g = outputs
            # _, idx = outputs[1].max(dim=2)
            # idx = idx.data.numpy()[0]
            # print(dictacgt[idx+1])
            # print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
            # loss_show.update(loss.item())


            _, pred = torch.max(outputs, 1)  # pred 取 0,1,2,3
            _, label = torch.max(labels, 1)  #
            c = (pred == label)  # 里面有多少true和false，
            right = torch.sum(c).item()
            accurate =100 *  right / batchsize
            accurate_show.update(accurate)
            print('[%d/15] loss = %.3f, acc:%.2f %%' % (epoch + 1, loss.item(),accurate))
        #print("total accuray:%.2f %%" % (accurate_show.avg))
        print('Epoch [%d/15] loss = %.3f, total acc: %.3f' % (epoch + 1, loss_show.avg,accurate_show.avg))
        loss_show.reset()
        accurate_show.reset()

        #validation,就用原图做validation









if __name__=="__main__":
    main()