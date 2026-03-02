import numpy as np
import matplotlib.pyplot as plt

# 1、散点输入
data = np.array([[0.8, 0], [1.1, 0], [1.7, 0], [3.2, 1], [3.7, 1], [4.0, 1], [4.2, 1]])

# 提取x 和y的值
x_data = data[:, 0]
y_data = data[:, 1]

# 转成Numpy数组
x_train = np.array(x_data)
y_train = np.array(y_data)


# 2、前向计算，意义不大（数学的方式无需使用前向计算）
# 3、激活函数的引入
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 4、参数的初始化
w = 0
b = 0
learning_rate = 0.5

# 构建两个图
fig, (ax1, ax2) = plt.subplots(2, 1)
epoch_list = []
loss_list = []

# 5、计算损失函数，（在数学的代码中无用）
# 6、开始迭代
epoches = 2000
for epoch in range(1, epoches + 1):
    pass
    # 7、反向传播计算，w新=w旧-lr*G
    z = w * x_train + b
    a = sigmoid(z)
    deda = -2 * (y_train - a)
    dadz = a * (1 - a)
    # 计算w的梯度
    dzdw = x_train
    gradient_w = np.mean(deda * dadz * dzdw)

    # 计算b的梯度
    dzdb = 1
    gradient_b = np.mean(deda * dadz * dzdb)

    # 更新参数，w新 = w旧 - lr * G
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

    # 计算损失
    z = w * x_train + b
    a = sigmoid(z)
    loss = np.mean((y_train - a) ** 2)

    epoch_list.append(epoch)
    loss_list.append(loss)

    # 8、显示频率设置
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoch:{epoch}, loss:{loss}")
        # 9、显示图像
        x_min = x_data.min()
        x_max = x_data.max()
        x_values = np.linspace(x_min, x_max, int((x_max - x_min) * 10))
        y_values = np.round(sigmoid(w * x_values + b), 3)

        ax1.clear()
        ax1.scatter(x_data, y_data)
        ax1.plot(x_values, y_values, c="r")

        ax2.clear()
        ax2.plot(epoch_list, loss_list, c="g")
        plt.pause(1)
plt.show()