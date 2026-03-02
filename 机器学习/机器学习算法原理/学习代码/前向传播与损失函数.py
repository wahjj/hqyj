import numpy as np
import matplotlib.pyplot as plt

# 1、数据集的输入
data = [[0.8, 1.0], [1.7, 0.9], [2.7, 2.4], [3.2, 2.9], [3.7, 2.8], [4.2, 3.8], [4.2, 2.7]]

data = np.array(data)

x_data = data[:, 0]
y_data = data[: ,1]

w = 0
b = 0
y_hat = w * x_data + b

e = y_data - y_hat
print(f"单点误差e:{e}")

e_bar = np.mean((y_data - y_hat) ** 2)
print(f"均方误差e_bar:{e_bar}")

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.set_xlim(0, 5)
ax1.set_ylim(0, 6)
ax1.set_xlabel("x axis label")
ax1.set_ylabel("y axis label")

ax1.scatter(x_data, y_data, color="b")

y_lower = w * 0 + b
y_upper = w * 5 + b
ax1.plot([0, 5], [y_lower, y_upper], color="r", linewidth=3)

for x, y_true, y_pre in zip(x_data, y_data, y_hat):
    ax1.plot([x, x], [y_true, y_pre], color="g", linestyle="--")

w_values = np.linspace(0, 3, 100)
e_values = [np.mean((y_data - (w_value * x_data + b)) ** 2) for w_value in w_values]
ax2.plot(w_values, e_values, color="g", linewidth=3)

ax2.plot(w, e_bar, marker="o", color="r")
ax2.set_xlabel("w axis label")
ax2.set_ylabel("e axis label")

plt.show()