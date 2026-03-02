import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# 读取数据
# header=None: 表示读取的文件没有头
dataset = pd.read_csv('../dataset/spambase.data', header=None)

# 查看读取进来的数据的前几行，确保数据集加载正确
# print(dataset.head())


# 数据预处理
# 提取特征数据
X = dataset.iloc[:, :-1]
# 提取标签数据
y = dataset.iloc[:, -1]

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 创建标准化对象，对原始数据进行数据标准化操作
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# 将标准化后的训练集与测试集转换为tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)



# DataLoader: 可以让我们高效的从数据集中批量加载数据   batch
# 使用之前，先将训练集特征张量与标签张量组合在一起 TensorDataset类
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# 创建DataLoader，用于批量加载数据，我们可以自定义每个批次加载多少样本，batch_size ，shuffle
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# 模型训练

# 定义逻辑回归模型
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# 获取训练集数组的特征张量的数量
input_size = X_train_tensor.shape[1]
# 实例化模型
model = LogisticRegression(input_size)

# 定义损失函数和优化器
# 使用二元交叉熵损失函数，用于二分类问题
criterion = nn.BCELoss()
# 使用Adam优化器，用于更新模型的参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 定义训练的次数epoch
num_epochs = 100
for epoch in range(num_epochs):
    # 将模型设置为训练模式
    model.train()

    # 总损失
    total_loss = 0

    # 遍历训练数据加载器，每次获取一个批次的输入数据和标签
    for inputs, labels in train_loader:
        # 清空优化器的梯度
        optimizer.zero_grad()
        # 前向传播，计算模型的输出
        outputs = model(inputs)

        # print(outputs.shape)
        # print(labels.shape)

        # 计算损失
        loss = criterion(outputs, labels.view(-1, 1))

        # 反向传播，计算梯度
        loss.backward()

        # 更新模型参数
        optimizer.step()

        # 将本次损失累加到总损失中
        total_loss += loss

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)

    # 每10个epoch打印一次信息
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1 }/{num_epochs}], Loss: {avg_loss:.4f}')

# 模型评估
with torch.no_grad():
    # 训练完毕之后，将模型设置为评估模式
    model.eval()
    # 使用测试集进行预测
    test_outputs = model(X_test_tensor)

    # 使用ROC曲线评估该模型
    fpr, tpr, threshold = roc_curve(y_test.values, test_outputs.numpy())

    # 计算ROC曲线下的面积（AUC），AUC取值在0到1之间，值越接近1，表示模型的性能越好
    roc_auc = auc(fpr, tpr)



# 可视化
plt.figure(figsize=(10, 6))
# 绘制ROC曲线
plt.plot(fpr, tpr, color='red', lw=2, label=f'AUC = {roc_auc:.2f}')

# 绘制一条对角线
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

# 设置x轴标签
plt.xlabel('False Positive Rate')
# 设置y轴标签
plt.ylabel('True Positive Rate')
# 设置标题
plt.title('Receiver Operating Characteristic(ROC) Curve')
# 显示图例
plt.legend()
# 显示图像
plt.show()
