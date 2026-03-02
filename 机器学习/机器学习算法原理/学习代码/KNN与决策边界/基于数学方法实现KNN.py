import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# 1、定义数据集
# 定义三个点集
point1 = [[7.7, 6.1], [3.1, 5.9], [8.6, 8.8], [9.5, 7.3], [3.9, 7.4], [5.0, 5.3], [1.0, 7.3]]
point2 = [[0.2, 2.2], [4.5, 4.1], [0.5, 1.1], [2.7, 3.0], [4.7, 0.2], [2.9, 3.3], [7.3, 7.9]]
point3 = [[9.2, 0.7], [9.2, 2.1], [7.3, 4.5], [8.9, 2.9], [9.5, 3.7], [7.7, 3.7], [9.4, 2.4]]

np_train_data = np.concatenate((point1,point2,point3))

np_train_label = np.array([0] * len(point1) + [1] * len(point2) + [2] * len(point3))

predict_point = np.array([3.5,4.2])

K = 3

distances = np.sqrt(np.sum((predict_point - np_train_data) ** 2,axis=1))

index = np.argsort(distances)

nearest_index = index[:K]

nearest_points = []
nearest_distances = []
nearest_label = []

for i in nearest_index:
    nearest_points.append(np_train_data[i])
    nearest_distances.append(distances[i])
    nearest_label.append(np_train_label[i])

counter = Counter(nearest_label)
print("数据点的label为：", counter.most_common()[0][0])

plt.xlabel("x axis label")
plt.ylabel("y axis label")
plt.scatter(np_train_data[np_train_label == 0, 0], np_train_data[np_train_label == 0, 1], marker="*")
plt.scatter(np_train_data[np_train_label == 1, 0], np_train_data[np_train_label == 1, 1], marker="^")
plt.scatter(np_train_data[np_train_label == 2, 0], np_train_data[np_train_label == 2, 1], marker="s")

plt.scatter(predict_point[0], predict_point[1], marker="o")

for i in range(K):
    plt.scatter([predict_point[0], nearest_points[i][0]],[predict_point[1], nearest_points[i][1]])
    plt.annotate(f"{nearest_distances[i]:2.2f}",
                 xy=((predict_point[0] + nearest_points[i][0]) / 2, (predict_point[1] + nearest_points[i][1]) / 2)
                 )

plt.show()