
# 导入所需的库
import os
import random

# 导入数据处理和可视化库
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 导入深度学习框架 PyTorch 相关库
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


# 设置随机种子以保证结果的可重复性
def setup_seed(seed):
    np.random.seed(seed)  # 设置 Numpy 随机种子
    random.seed(seed)  # 设置 Python 内置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 哈希种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种子
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # 关闭 cudnn 加速
        torch.backends.cudnn.deterministic = True  # 设置 cudnn 为确定性算法


# 设置随机种子
setup_seed(0)

# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # 使用 CPU
    print("CUDA is not available. Using CPU.")

# 设置 Pandas 显示选项，以便显示更多的列和行内容
# pd.set_option('display.max_columns', 1000)
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_colwidth', 1000)

# 读取训练数据集
train_data = pd.read_csv("../dataset/train_V2.csv")
print(len(train_data))

# 打印数据集的最后几行和信息摘要
print(train_data.tail())  # 打印数据集的最后几行
print(train_data.info())  # 打印数据集的信息摘要



# 检查数据集中的缺失值情况
# 统计数据集中每列的缺失值数量
# train_data.isnull() 是 pandas 中 DataFrame 对象的一个方法调用
# 该方法会对 train_data 数据框中的每个元素进行检查
# 如果元素是缺失值（例如 NaN 或 None），则该位置会被标记为 True
# 否则标记为 False，最终返回一个与 train_data 形状相同的布尔型 DataFrame
# 其中元素为布尔值，True 代表该位置是缺失值，False 代表不是缺失值
# .sum() 是对布尔型 DataFrame 按列进行求和操作
# 在 Python 里，布尔值 True 可以被视为 1，False 可以被视为 0
# 所以求和的结果就是每列中 True 的数量，也就是每列中缺失值的数量
# 最终得到一个 Series 对象，索引是列名，值是对应列的缺失值数量
print(train_data.isnull().sum())
train_data = train_data.dropna(subset=['winPlacePerc'])


# 查看删除缺失值后的数据基本信息，确认是否删除成功
print(train_data.info())
print(train_data.isnull().sum())


# 对'matchId'字段进行分组求和，以计算每次比赛的人数
# 1. 按比赛 ID 分组并统计每组的记录数量
# train_data.groupby('matchId')：使用 pandas 的 groupby 方法，按照 'matchId' 列对 train_data 数据集进行分组。
# 这一步会将具有相同 'matchId' 的数据行归为一组，每个组代表一场比赛。
# ['matchId']：从分组后的数据中选择 'matchId' 列，明确后续操作针对该列。
# transform('count')：对每个分组应用 'count' 函数进行计数。
# 与直接使用 'count' 不同，transform 会将每个分组的计数结果广播到该分组对应的每一行数据上，
# 最终返回一个与 train_data 行数相同的 Series 对象，其中每个元素表示该行所属比赛的记录数量，也就是参与该比赛的人数。
count = train_data.groupby('matchId')['matchId'].transform('count')


'''
这个打印结果是对 train_data 数据集按 matchId 分组并对每组中 matchId 列元素计数后得到的结果，
每一行代表原数据集中对应行所在分组的计数情况，下面详细解释：
索引列：左侧的数字（0、1、2 等，一直到 4446965）是 count 这个 Series 对象的索引，
它对应着 train_data 数据集中的行索引。也就是说，索引 0 对应的计数 96，表示 train_data 中索引为 0 的那一行数据，
其所在的以 matchId 分组的组内有 96 条记录；同理，索引 4446965 对应的计数 98，表示 train_data 中索引为 4446965 的那一行数据，
其所在组内有 98 条记录 。
计数列：右侧的数字（96、91、98 等）是每个分组中 matchId 列元素的数量，
即参与对应比赛的人数（因为是按 matchId 分组统计 matchId 列元素个数，在这个场景下可以理解为比赛人数）。
从结果可以看出，不同分组（不同比赛场次）的人数有所不同，比如有的比赛有 91 人参与，有的有 98 人参与。
其他信息：
Name: matchId：表示这个 Series 对象的名称是 matchId，这里的名称主要用于标识数据的含义，在后续与其他数据交互或操作时可能会用到。
Length: 4446966：说明 count 这个 Series 对象包含 4446966 个元素，这与 train_data 数据集的行数是一致的，因为 transform 方法会将每个分组的计数结果广播到该分组对应的每一行数据上。
dtype: int64：表示 count 这个 Series 对象中元素的数据类型是 64 位整数。
'''
print('count:', count)


# 2. 将统计结果添加到原数据集中
# train_data['playersJoined']：在 train_data 数据集中创建一个新的列，列名为 'playersJoined'。
# count：将前面计算得到的每个比赛的参与人数 Series 对象赋值给新列 'playersJoined'。
# 这样，原数据集中的每一行都有了对应的比赛参与人数信息。
# 3. 对 'playersJoined' 列进行排序并打印结果
# train_data["playersJoined"]：从 train_data 数据集中选择 'playersJoined' 列。
# sort_values()：对选择的 'playersJoined' 列进行排序，默认是升序排序。
# 排序后的数据有助于我们观察不同比赛参与人数的分布情况，例如是否存在参与人数较少或较多的异常比赛。
# print()：将排序后的结果打印输出，方便查看。
train_data['playersJoined'] = count
print(train_data["playersJoined"].sort_values())
print(train_data.head())


# plt.figure(0)
# sns.countplot(train_data['playersJoined'])
# plt.title('playersJoined')
# plt.grid()
# plt.xticks(rotation=90)


# 选取 train_data["playersJoined"] 等于 100 的数据，即只选取比赛人数为100的作为训练数据
selected_data = train_data[train_data["playersJoined"] == 100]

# 确认是否选取了正确的数据
print(selected_data.head())
print(len(selected_data))

# 将数据集划分为特征集（X）和目标集（y），并对 matchType 列进行独热编码处理
X = selected_data.drop(['Id', 'groupId', 'matchId'], axis=1)
y = selected_data['winPlacePerc']
X_encoded = pd.get_dummies(X, columns=['matchType'])

# 将数据集按比例划分为训练集和测试集
train_ratio = 0.8
X_train = X_encoded[:int(train_ratio * len(selected_data))]
X_test = X_encoded[int(train_ratio * len(selected_data)):]
y_train = y[:int(train_ratio * len(selected_data))]
y_test = y[int(train_ratio * len(selected_data)):]

# 使用标准化进行特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# 定义一个简单的 DNN 模型
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 实例化模型
input_size = X_train_tensor.shape[1]
hidden_size = 128
output_size = 1
model = DNN(input_size, hidden_size, output_size).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor.to(device))
    test_loss = criterion(predictions, y_test_tensor.to(device))

# 将预测值和目标值转换为 NumPy 数组
predictions = predictions.cpu().numpy()
y_test_numpy = y_test_tensor.cpu().numpy()
#
print('MSE', test_loss)

# 绘制结果
plt.figure(1)
plt.scatter(y_test_numpy, predictions, color='blue')
plt.plot([min(y_test_numpy), max(y_test_numpy)], [min(y_test_numpy), max(y_test_numpy)], linestyle='--', color='red',
         linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Regression Results')

# 绘制实际值和预测值的曲线
plt.figure(2)
plt.plot(y_test_numpy[-100:], label='Actual Values', marker='o')
plt.plot(predictions[-100:], label='Predicted Values', marker='x')
plt.xlabel('Sample Index (Sorted)')
plt.ylabel('Values')
plt.title('Actual vs Predicted Values in Linear Regression')
plt.legend()
plt.show()
