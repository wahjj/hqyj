
import pandas as pd
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader



# 设置随机种子保证结果的可重复性
def setup_seed(seed):
    # 设置 Numpy 随机数种子，确保Numpy生成的随机数序列一致
    np.random.seed(seed)

    # 设置Python内置随机数种子，保证Python内置的随机函数生成的随机数一致
    random.seed(seed)

    # 设置Python哈希种子，避免不同运行环境下哈希结果不同，影响随机数生成
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 设置PyTorch 随机种子，使PyTorch生成的随机数序列可以重复
    torch.manual_seed(seed)

    # 检查是否有可用的CUDA设备（GPU）
    if torch.cuda.is_available():
        # 设置 CUDA 随机种子，保证在GPU上的随即操作可重复
        torch.cuda.manual_seed(seed)
        # 为所有 GPU 设置随机种子
        torch.cuda.manual_seed_all(seed)
        # 关闭 cudnn 自动寻找最优算法加速的功能，保证结果可重复
        torch.backends.cudnn.benchmark = False
        # 设置 cudnn 为确定性算法，确保每次运行结果一致
        torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    device = 'cuda'
    print('CUDA is useful!!')
else:
    device = 'cpu'
    print('CUDA is not useful!!')

setup_seed(0)

# 设置 pandas 显示选项，以便显示更多的列和行的内容
# 最多显示1000列
pd.set_option('display.max_columns', 1000)
# 显示宽度为1000
pd.set_option('display.width', 1000)
# 每列最多显示1000个字符
pd.set_option('display.max_colwidth', 1000)


# 读取数据集，需要注意的是；编码格式必须为 big5
train_data = pd.read_csv('../dataset/train.csv', encoding='big5')
# 查看读取进来的前5行数据，确保数据被正确读取
# print(train_data.head())

# 打印数据集的信息，查看数据集的情况
# print(train_data.info())


# 选取从第3列开始到最后的所有列作为特征数据
train_data = train_data.iloc[:, 3:]

# 将数组中值为NR的元素替换为0
train_data[train_data == 'NR'] = 0

# 将train_data转换为Numpy数组
numpy_data = train_data.to_numpy()

# # 检查数据集中缺失值情况
# print(train_data.isnull().sum())

# 创建一个列表，用来存储拆分后的数据
datas = []

# 按照 步长为18 分割数据
for i in range(0, 4320, 18):
    datas.append(numpy_data[i:i+18, :])

# print(datas)

# 将datas 转换为Numpy数组
datas_array = np.array(datas, dtype=float)

# print(datas_array.shape)

# 对数据进行维度变换和重塑，转为DataFrame格式
train_data = pd.DataFrame(datas_array.transpose(1, 0, 2).reshape(18, -1).T)

# print(test.shape)

# 计算特征相关性矩阵
corr = train_data.corr()

# 绘制相关性热图
plt.figure(0)

# seaborn库：基于matplotlib的高级可视化库
# heatmap()：热力图，用于展示矩阵数据，通过颜色的深浅表示数值大小，常用于展示相关性矩阵、混淆矩阵等内容，可以帮助用户快速
# 发现数据之间的关系
sns.heatmap(corr, annot=True)


# 从相关性矩阵里筛选比较重要的特征
important_features = []
for i in range(len(corr.columns)):
    # 选择与第9列相关性系数绝对值大于0.2的特征
    if abs(corr.iloc[i, 9]) > 0.2:
        important_features.append(corr.columns[i])

# print('比较重要的特征：', important_features)


# 确定目标特征所在列之后，就要去将数据集划分特征和目标了
# 选取重要特征，但是要排除掉目标本身，剩下的所有的重要特征就是我们的目标特征了
X = train_data[important_features].drop(9, axis=1)
# 选取第9列作为目标
y = train_data[9]


# 划分训练集和测试集
train_ratio = 0.8

# 训练集特征
X_train = X[:int(train_ratio * len(train_data))]
# 测试集特征
X_test = X[int(train_ratio * len(train_data)):]
# 训练集标签
y_train = y[:int(train_ratio * len(train_data))]
# 测试集标签
y_test = y[int(train_ratio * len(train_data)):]


# print(type(y_train))

# 使用标准化进行特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 将数据转换为PyTorch的张量
# 训练集的特征张量表示
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
# 训练集的标签张量表示
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
# 测试集的特征张量表示
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
# 测试集的标签张量表示
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# 使用DataLoader去加载数据集
# 创建TensorDataset对象，将训练集特征张量与标签一一组合在一起
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# 创建DataLoader对象，用于批量加载数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



# 构建DNN模型
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # 调用父类的构造函数
        super(DNN, self).__init__()
        # 定义第一个Linear层，输入就是input_size，输出维度就是 hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 定义第二个Linear层，输入是hidden_size，输出维度是 hidden_size
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 定义第三个Linear层，输入是hidden_size，输出维度是 hidden_size
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # 定义第四个Linear层，输入是hidden_size，输出维度是 output_size
        self.fc4 = nn.Linear(hidden_size, output_size)
        # 定义激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入数据经过第一个Linear和ReLU激活
        x = self.relu(self.fc1(x))
        # 输入数据经过第二个Linear和ReLU激活
        x = self.relu(self.fc2(x))
        # 输入数据经过第三个Linear和ReLU激活
        x = self.relu(self.fc3(x))
        # 输入数据经过第四个Linear进行输出
        x = self.fc4(x)
        return x

# 实例化模型
# 需要获取input_size，hidden_size, outputs_size
# 将输入特征的列数作为输入的维度
input_size = X_train_tensor.shape[1]

# hidden_size：自由发挥
hidden_size = 256

# output_size 标签的维度
output_size = 1

# 模型实例化成功
model = DNN(input_size, hidden_size, output_size)

# 将模型放到device上
model.to(device)


# 定义损失函数，使用均方误差损失函数，因为是回归任务
criterion = nn.MSELoss()
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)
# 定义训练的epoch
num_epochs = 500


for epoch in range(num_epochs):
    # 将模型设置为训练模式
    model.train()

    # 初始化总损失
    total_loss = 0

    # 循环加载dataLoader，每次获取一个批次的数据和标签
    for inputs, labels in train_loader:
        # 清空优化器梯度
        optimizer.zero_grad()

        # 将inputs和labels放到device上，即将输入数据放到指定的设备上
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播，计算输出
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()

        # 更新参数
        optimizer.step()

        # 累加当前批次的损失
        total_loss += loss

    # 计算本次epoch的平均损失
    avg_loss = total_loss / len(train_loader)

    # 每10个epoch打印一次当前模型的训练情况
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')


# 模型的评估
with torch.no_grad():
    # 将模型设置为评估模式
    model.eval()
    # 将测试集放到模型上计算输出
    predict_test = model(X_test_tensor.to(device))
    # 计算测试集的损失
    test_loss = criterion(predict_test, y_test_tensor.to(device))


# 将预测值和目标值转换为Numpy数组
# 将预测值从GPU移动到CPU上再去转换为Numpy
predictions = predict_test.cpu().numpy()
# 将测试集目标值从CPU上移动到CPU上并转换为Numpy
y_test_numpy = y_test_tensor.cpu().numpy()

# 打印测试集的损失
print('MSE:', test_loss.item())

# 绘制结果，创建一个新的图形窗口
plt.figure(1)
# 绘制散点图，用来显示实际值和预测值的关系
plt.scatter(y_test_numpy, predictions, color='blue')
# 绘制对角线，用来对比
plt.plot([min(y_test_numpy), max(y_test_numpy)], [min(y_test_numpy), max(y_test_numpy)], linestyle='--', color='red', lw=2)
# 设置x轴标签
plt.xlabel('Actual Values')
# 设置y轴标签
plt.ylabel('Predicted Values')
# 设置标题
plt.title('Regression Results')


# 绘制实际值和预测值的曲线
# 创建一个新的图形窗口
plt.figure(2)
# 绘制实际值的曲线，取最后100个样本
plt.plot(y_test_tensor[-100:], label='Actual Values', marker='o')
# 绘制预测值的曲线，取最后100个样本
plt.plot(predictions[-100:], label='Predicted Values', marker='*')
# 设置x轴标签
plt.xlabel('Sample Index')
# 设置y轴标签
plt.ylabel('Values')
# 设置标题
plt.title('Actual vs Predicted Values in Linear Regression')

plt.show()