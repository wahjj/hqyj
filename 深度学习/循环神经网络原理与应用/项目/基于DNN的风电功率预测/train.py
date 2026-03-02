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


setup_seed(0)

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
    def __init__(self, csv_path, input_len):
        self.input_len = input_len
        self.data = pd.read_csv(csv_path)
        self.data["功率(kW)"] = np.minimum(self.data["功率(kW)"], 1500)
        # 实例化归一化的类
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        # reshape(-1, 1)之后得到[samples, 1]
        self.data["power_normalized"] = self.scaler.fit_transform(self.data["功率(kW)"].values.reshape(-1, 1))

    def __len__(self):
        # 返回数据集的长度
        return len(self.data) - self.input_len

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.input_len
        feature = self.data["power_normalized"].values[start_idx:end_idx]
        target = self.data["power_normalized"].values[end_idx:end_idx+1]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

input_len = 6
power_dataset = PowerData("A01.csv", input_len)

# 计算比例并划分每个数据集的内容
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
train_size = int(train_ratio * len(power_dataset))
val_size = int(val_ratio * len(power_dataset))
test_size = test_ratio * len(power_dataset)
# [0,1,2,3,...,dataset_size-1]
indices = list(range(len(power_dataset)))
train_dataset = Subset(power_dataset, indices[:train_size])
val_dataset = Subset(power_dataset, indices[train_size:train_size + val_size])
test_dataset = Subset(power_dataset, indices[train_size + val_size:])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

class DNN(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=1):
        super(DNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


model = DNN().to(device)
cri = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

epochs = 20
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for batch_feature, batch_target in train_dataloader:
        batch_feature, batch_target = batch_feature.to(device), batch_target.to(device)
        # 前向传播
        y_pred = model(batch_feature)
        # y_pred [bs, 1] -> [bs,]
        loss = cri(y_pred.squeeze(1), batch_target.view(-1))
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
            total_loss += cri(y_pred.squeeze(1), batch_target.view(-1)).item()
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
        predict_list.append(y_pred.squeeze(1).item())
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

