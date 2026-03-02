import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据集
# 使用pandas的read_csv函数读取名为'abalone.data'的数据集文件
# 这里假设数据文件位于'../dataset/'目录下，读取后可通过打印头部数据初步查看数据内容和格式
data = pd.read_csv('../dataset/abalone.data')
print(data.head())

# 添加列名
# 定义一个包含各列名称的列表，为数据框中的列赋予有意义的名称
# 使得后续数据处理和分析时更清晰明确各列代表的含义
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
data.columns = column_names

# 对 'Sex' 列进行独热编码
# 因为 'Sex' 列可能是分类变量（比如包含不同的性别类别），独热编码会将其转换为多个二进制列
# 例如原本的 'Sex' 列有 'F'、'M'、'I' 等类别，编码后会生成 'Sex_F'、'Sex_M'、'Sex_I' 等新列，方便模型处理分类特征
data = pd.get_dummies(data, columns=['Sex'])
print(data.keys())

# 提取特征和目标变量
# 从处理后的数据框中选取要作为模型输入特征的列，组成特征矩阵X
# 这里包含了独热编码后的性别相关列以及其他如长度、直径等连续特征列
X = data[['Sex_F', 'Sex_M', 'Sex_I', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']]
# 选取 'Rings' 列作为目标变量，即模型要预测的对象，通常代表了鲍鱼的年龄相关信息（可能通过年轮等方式衡量）
y = data['Rings']

# 将数据分为训练集和测试集
# 使用sklearn的train_test_split函数按照指定的测试集比例（test_size=0.2，表示20%的数据作为测试集）
# 和随机种子（random_state=42，保证每次划分结果一致，便于复现和对比不同实验情况）来划分数据集
# 划分后得到训练集特征X_train、测试集特征X_test、训练集目标y_train、测试集目标y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对数据进行标准化
# 创建StandardScaler对象，用于对数据进行标准化处理，使其具有均值为0，方差为1的分布特点
# 先在训练集上拟合标准化器（计算均值、方差等统计量），然后用拟合好的标准化器对训练集和测试集分别进行转换
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将数据转换为 PyTorch 张量
# 将标准化后的训练集特征、训练集目标、测试集特征、测试集目标都转换为PyTorch的张量格式
# 并指定数据类型为float32，这是因为PyTorch模型中的计算通常要求数据为特定的张量类型，方便后续运算
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# 定义线性回归模型
# 自定义的线性回归模型类，继承自nn.Module，这是PyTorch中构建神经网络模型的基类
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        # 定义一个线性层，输入维度为input_size（即特征数量），输出维度为1，对应要预测的目标变量维度（这里是预测的环数）
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # 前向传播函数，定义了数据如何通过模型进行计算，这里直接将输入数据x传入线性层并返回其输出结果
        return self.linear(x)


# 实例化模型
# 根据训练集特征张量的列数确定输入维度，即特征数量，以此实例化线性回归模型
input_size = X_train_tensor.shape[1]
model = LinearRegressionModel(input_size)

# 定义损失函数和优化器
# 使用均方误差损失函数（MSELoss），它可以衡量预测值和真实值之间的差异程度，是回归问题常用的损失函数之一
criterion = nn.MSELoss()
# 使用Adam优化器，传入模型的可学习参数（也就是模型中线性层的权重等参数），并指定学习率为0.1
# 优化器的作用是根据损失函数计算得到的梯度信息来更新模型参数，以尝试减小损失函数的值
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    # 前向传播
    # 将训练集特征张量传入模型，通过模型的前向传播计算得到预测值
    outputs = model(X_train_tensor)
    # 计算预测值和真实训练集目标值之间的损失，调用之前定义的损失函数来完成计算
    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    # 计算损失关于模型参数的梯度，PyTorch会自动根据构建的计算图来完成这一复杂的求导过程
    loss.backward()
    # 根据计算得到的梯度，使用优化器来更新模型的参数，按照学习率等设置调整参数值，使得模型朝着损失减小的方向优化
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    # 在测试集上进行预测，将测试集特征张量传入模型得到预测值
    predictions = model(X_test_tensor)
    # 计算测试集上的损失，用于评估模型在未见过的数据（测试集）上的表现，同样调用损失函数来计算
    test_loss = criterion(predictions, y_test_tensor)

# 将预测值和目标值转换为 NumPy 数组
# 把PyTorch张量形式的预测值和测试集目标值转换为NumPy数组，方便后续使用matplotlib等库进行数据可视化操作
predictions = predictions.numpy()
y_test_numpy = y_test_tensor.numpy()

# 绘制结果
# 使用matplotlib绘制散点图展示预测值和实际值的关系
# 蓝色散点表示实际值和预测值对应的各个点，直观呈现模型预测效果与实际情况的差异
plt.scatter(y_test_numpy, predictions, color='blue')
# 绘制一条对角线（虚线红色，线宽为2），代表理想情况下预测值和实际值完全相等的情况，用于对比参考
plt.plot([min(y_test_numpy), max(y_test_numpy)], [min(y_test_numpy), max(y_test_numpy)], linestyle='--', color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Regression Results')
plt.show()