import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import numpy as np

# 1、散点输入
data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7], [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]
# 转换成Numpy
data = np.array(data)
# 提取x和y
x_data = data[:, 0]
y_data = data[:, 1]

# 将x_data 和 y_data 转换成 tensor
x_train = torch.tensor(x_data, dtype=torch.float32)
y_train = torch.tensor(y_data, dtype=torch.float32)

# 设置随机数种子
seed = 42
torch.manual_seed(seed)

# 2、定义前向模型
model = nn.Linear(1, 1)  # 输入特征是1，输出特征是1

# 1)nn.Sequential是pytorch的一个模块容器，按顺序组合多个网络层
# nn.Sequential默认带forward方法
# forward方法会定义模型的前向传播逻辑，给定输入，经过逻辑，得到输出
# model = nn.Sequential(nn.Linear(1, 1))

# 2)nn.ModuleList

# model = nn.ModuleList([nn.Linear(1, 1)])
# class LinearModel(nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.layers = nn.ModuleList([nn.Linear(1, 1)])
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
#
# model = LinearModel()

# 3)nn.ModuleDict
# nn.ModuleDict：可以给每个层自定义名字
#  nn.ModuleList：不可以给每个层自定义名字
# model = nn.ModuleDict({"linear": nn.Linear(1, 1)})
# class LinearModel(nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.layers = nn.ModuleDict({"linear": nn.Linear(1, 1)})
#
#     def forward(self, x):
#         for layer in self.layers.values():
#             x = layer(x)
#         return x
#
# model = LinearModel()

"""
实际上最常用的
"""
# class LinearModel(nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = nn.Linear(1, 1)
#         self.linear2 = nn.Linear(2, 1)
#
#     def forward(self, x):
#         x = self.linear(x)
#         return x
#
# model = LinearModel()


# 3、定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# 用于绘制三维图形用
def loss_function(X, Y, w, b):
    predicted_y = np.dot(X, w) + b
    total_loss = np.mean((2 * (predicted_y - Y) ** 2))
    return total_loss

# 用来记录梯度
gd_path = []

# 构建网格点
w_values = np.linspace(-20, 80, 100)
b_values = np.linspace(-20, 80, 100)
W, B = np.meshgrid(w_values, b_values)
loss_values = np.zeros_like(W)

for i, w in enumerate(w_values):
    for j, b in enumerate(b_values):
        loss_values[j, i] = loss_function(x_data, y_data, w, b)

# 创建图形对象和子图布局
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 2)

# 左上格子
ax2 = fig.add_subplot(gs[0, 0])
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("Data")

# 左下格子
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlabel("w")
ax3.set_ylabel("b")
ax3.set_title("Contour Plot")

# 整个右侧格子
ax1 = fig.add_subplot(gs[:, 1], projection='3d')
ax1.plot_surface(W, B, loss_values, cmap='viridis', alpha=0.8)
ax1.set_xlabel('w')
ax1.set_ylabel('b')
ax1.set_zlabel('Loss')
ax1.set_title("Surface Plot")

# 获取模型初始化过的参数
w = float(model.weight)
b = float(model.bias)


# 4、开始迭代
epoches = 500
for n in range(1, epoches+1):
    gd_path.append((w, b))

    y_hat = model(x_train.unsqueeze(1))

    loss = criterion(y_hat.squeeze(1), y_train) # 计算损失

    optimizer.zero_grad() # 作用：清空之前存储在优化器中的梯度

    loss.backward() # 计算损失函数关于模型参数的梯度

    optimizer.step() # 根据优化算法去更新参数

    w = float(model.weight)
    b = float(model.bias)

    if n % 10 == 0 or n == 1:
        print(f"epoches:{n}, loss:{loss}")

        # 根据当前参数拟合直线
        x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
        y_line = np.dot(x_line, w) + b

        # 更新子图 1 数据并绘制
        ax2.clear()
        ax2.scatter(x_data, y_data)
        ax2.plot(x_line, y_line, '-')
        ax2.set_title(f"Linear Regression: w={w}, b={b}")
        # 绘制当前w和b的位置
        ax1.scatter(w, b, loss_function(x_data, y_data, w, b), c='black', s=20)

        # 绘制俯视图等高线
        ax3.clear()
        ax3.contourf(W, B, loss_values, levels=20, cmap='viridis')
        ax3.scatter(w, b, c='black', s=20)

        # 绘制梯度下降路径
        if len(gd_path) > 0:
            gd_w, gd_b = zip(*gd_path)
            ax1.plot(gd_w, gd_b,
                     [loss_function(x_data, y_data, np.array(gd_w[i]), np.array(gd_b[i])) for i in range(len(gd_w))], c='black')
            ax3.plot(gd_w, gd_b)
        plt.pause(1)

plt.show()