import numpy as np
import matplotlib.pyplot as plt


def pdf(x, mean, cov):
    # 获取均值向量的长度，即特征的数量
    n = len(mean)
    # 计算PDF的系数部分
    coeff = 1 / ((2 * np.pi) ** (n/2) * np.sqrt(np.linalg.det(cov)))
    # 计算PDF的指数部分
    exponet = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    return coeff * np.exp(exponet)


# 1.散点输入
class1_points = np.array([[1.9, 1.2],
                          [1.5, 2.1],
                          [1.9, 0.5],
                          [1.5, 0.9],
                          [0.9, 1.2],
                          [1.1, 1.7],
                          [1.4, 1.1]])

class2_points = np.array([[3.2, 3.2],
                          [3.7, 2.9],
                          [3.2, 2.6],
                          [1.7, 3.3],
                          [3.4, 2.6],
                          [4.1, 2.3],
                          [3.0, 2.9]])
# 合并数据集、创造标签
X = np.concatenate((class1_points, class2_points))
y = np.concatenate((np.zeros(len(class1_points)), np.ones(len(class2_points))))

# 2. 计算先验概率(每一个类别的数据在数据集中的比例)
prior_probabilities = [np.sum(y == 0) / len(y), np.sum(y == 1) / len(y)]
print(prior_probabilities)

# 3.计算高斯分布的概率密度函数
# 求解每个类别的均值
class_means = [np.mean(X[y == 0], axis=0), np.mean(X[y == 1], axis=0)]
# print(class_means)

# 求解每个类别的协方差矩阵
X_y_0 = X[y == 0].T
X_y_1 = X[y == 1].T
class_covs = [np.cov(X_y_0), np.cov(X_y_1)]

point = np.array([3, 1.5])

posterior_probabilities = []
for i in range(2):
    # 使用高斯分布的概率密度函数求解条件概率
    likelihood = pdf(point, class_means[i], class_covs[i])
    # 4.得到后验概率，比较大小，获得分类
    posterior_probabilities.append(prior_probabilities[i] * likelihood)

# 比较大小，获得分类
pre_class = np.argmax(posterior_probabilities)

# 5. 显示归属
print(f"点 {point} 属于类别 {pre_class}")

plt.scatter(class1_points[:, 0], class1_points[:, 1], c="blue", label="class 1")
plt.scatter(class2_points[:, 0], class2_points[:, 1], c="red", label="class 2")
plt.scatter(point[0], point[1], c="green", label="point")
plt.legend()
if pre_class == 0:
    plt.text(point[0] + 0.1, point[1] - 0.1, "class 1")
else:
    plt.text(point[0] + 0.1, point[1] - 0.1, "class 2")
plt.show()