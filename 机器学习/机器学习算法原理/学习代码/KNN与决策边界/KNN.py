from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# 1、定义数据集
# 定义三个点集
point1 = [[7.7, 6.1], [3.1, 5.9], [8.6, 8.8], [9.5, 7.3], [3.9, 7.4], [5.0, 5.3], [1.0, 7.3]]
point2 = [[0.2, 2.2], [4.5, 4.1], [0.5, 1.1], [2.7, 3.0], [4.7, 0.2], [2.9, 3.3], [7.3, 7.9]]
point3 = [[9.2, 0.7], [9.2, 2.1], [7.3, 4.5], [8.9, 2.9], [9.5, 3.7], [7.7, 3.7], [9.4, 2.4]]

np_train_data = np.concatenate((point1,point2,point3),axis=0)
# print(np_train_data)

np_train_label = np.array([0] * len(point1) + [1] * len(point2) + [2] * len(point3))

knn_clf = KNeighborsClassifier(3)

knn_clf.fit(np_train_data,np_train_label)

axis = [0,10,0,10]

x0,x1 = np.meshgrid(
np.linspace(axis[0],axis[1],100),
    np.linspace(axis[0],axis[1],100)
)

axis_xy = np.c_[x0.ravel(),x1.ravel()]

y_predict = knn_clf.predict(axis_xy)

y_predict = y_predict.reshape(x0.shape)

plt.contour(x0,x1,y_predict)

plt.scatter(np_train_data[np_train_label == 0, 0],np_train_data[np_train_label == 0, 1],marker="^")
plt.scatter(np_train_data[np_train_label == 1, 0],np_train_data[np_train_label == 1, 1],marker="^")
plt.scatter(np_train_data[np_train_label == 2, 0],np_train_data[np_train_label == 2, 1],marker="^")
plt.show()