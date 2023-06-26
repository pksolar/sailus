import torch
import torch.nn as nn

class DNA_Sequencer(nn.Module):
    def __init__(self):
        super(DNA_Sequencer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 3 * 3, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = x.view(-1, 32 * 3 * 3)
        x = self.fc(x)
        return x

model = DNA_Sequencer()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(100):
    # 获取训练数据
    inputs, labels = get_train_data()
    # 清空梯度
    optimizer.zero_grad()
    # 前向传播
    outputs = model(inputs)
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()

# 处理最终输出
outputs = model(inputs)
_, preds = torch.max(outputs, 1)
preds = preds.tolist()

# 将输出的数字转换为碱基
base_list = ['A', 'C', 'G', 'T', 'N']
preds = [base