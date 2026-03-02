import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# 定义随机种子，保证每次训练、预测结果固定
def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


setup_seed(1)

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    # 如果有 GPU，选择使用 GPU
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    # 如果没有 GPU，使用 CPU
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


class PowerData(Dataset):
    def __init__(self, csv_path, sequence_len):
        self.sequence_len = sequence_len  # 序列长度（用前20个点预测下1个）
        self.data = pd.read_csv(csv_path) # 读取功率数据CSV
        self.data["功率(kW)"] = np.minimum(self.data["功率(kW)"], 1500) # 功率上限截断（防止异常值）
        # 实例化归一化的类
        self.scaler = MinMaxScaler(feature_range=(-1, 1)) # 归一化到[-1,1]
        # reshape(-1, 1)之后得到[samples, 1]
        self.data["power_normalized"] = self.scaler.fit_transform(self.data["功率(kW)"].values.reshape(-1, 1))

    def __len__(self):
        # 返回数据集的长度
        return len(self.data) - self.sequence_len

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.sequence_len
        feature = self.data["power_normalized"].values[start_idx:end_idx]  # 前20个归一化功率（特征）
        target = self.data["power_normalized"].values[end_idx:end_idx+1]  # 第21个归一化功率（目标）
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
        # 特征：shape = [20]（长度为20的时序序列）；
        # 目标：shape = [1]（单个预测值）；
        # 数据类型：转为float32（PyTorch默认浮点类型，节省内存且兼容GPU）。

sequence_len = 20
power_dataset = PowerData("./A01.csv", sequence_len)

# 计算比例并划分每个数据集的内容
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
train_size = int(train_ratio * len(power_dataset)) #
val_size = int(val_ratio * len(power_dataset))
test_size = test_ratio * len(power_dataset)
# [0,1,2,3,...,dataset_size-1]
indices = list(range(len(power_dataset)))
train_dataset = Subset(power_dataset, indices[:train_size])
val_dataset = Subset(power_dataset, indices[train_size:train_size + val_size])
test_dataset = Subset(power_dataset, indices[train_size + val_size:])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False) #batch_size=1（测试集）：逐个样本预测，方便后续拼接结果。

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1):
        """
        初始化
        :param input_size: 输入数据的特征维度大小，每个时间步输入的特征向量的维度
        :param hidden_size:
        :param output_size:
        """
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print("----", x.shape)  # torch.Size([64, 20])
        x = x.unsqueeze(2)  # torch.Size([64, 20, 1])  # # [64, 20] → [64, 20, 1]：补充特征维度（input_size=1）
        w, h_n = self.rnn(x)  # h_n是最后一个时间步的隐藏状态  ## RNN前向计算
        # print("h_n:", h_n.shape)  # (num_layers*num_directions, batch, hidden_size)->[1, 64, 128]
        # 例如num_layers=2， [2, 64, 128]
        x = (self.linear2(h_n[-1])).squeeze(1)  # self.linear2的输出张量[64, 1]

        # x = F.relu(self.linear1(x))
        # x = self.linear2(x)
        return x

        # x.unsqueeze(2)：输入是[batch_size, seq_len]（64, 20），但RNN要求输入格式为[batch_size, seq_len, input_size]，因此补充第2维（特征维），变为(64, 20, 1)；
        #
        # self.rnn(x)：
        # 输出w：所有时间步的隐藏状态，shape = [64, 20, 128]；
        # 输出h_n：最后一个时间步的隐藏状态，shape = [1, 64, 128]（num_layers = 1，num_directions = 1）；
        #
        # h_n[-1]：取最后一层的隐藏状态（此处只有1层），shape = [64, 128]；
        #
        # linear2：将128维隐藏状态映射为1维预测值，shape = [64, 1]；
        #
        # squeeze(1)：去掉维度1，变为[64]（匹配目标值的shape）。


model = RNN().to(device)
cri = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

epochs = 100
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for batch_feature, batch_target in train_dataloader:
        batch_feature, batch_target = batch_feature.to(device), batch_target.to(device)
        # 前向传播
        y_pred = model(batch_feature)
        # y_pred [bs, 1] -> [bs,]
        loss = cri(y_pred, batch_target.view(-1))
        # 反向传播更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    train_loss = total_loss / len(train_dataloader)
    # 验证集的损失
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_feature, batch_target in val_dataloader:
            batch_feature, batch_target = batch_feature.to(device), batch_target.to(device)
            y_pred = model(batch_feature)
            total_loss += cri(y_pred, batch_target.view(-1)).item()
    val_loss = total_loss / len(val_dataloader)
    print(f"Epoch:[{epoch}/{epochs}], Train Loss: {train_loss:.4f}, Eval Loss: {val_loss:.4f}")

# 计算测试结果
model.eval()
predict_list = []
target_list = []
with torch.no_grad():
    for batch_feature, batch_target in test_dataloader:
        batch_feature, batch_target = batch_feature.to(device), batch_target.to(device)
        y_pred = model(batch_feature)
        predict_list.append(y_pred.item())
        target_list.append(batch_target.item())

# 将预测结果反归一化
predict_list = power_dataset.scaler.inverse_transform(np.array(predict_list).reshape(-1, 1))
target_list = power_dataset.scaler.inverse_transform(np.array(target_list).reshape(-1, 1))

plt.plot(target_list, label="True values")
plt.plot(predict_list, label="Predict values")
plt.xlabel("Time")
plt.ylabel("Power")
plt.legend()
plt.show()












