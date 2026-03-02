import numpy as np

print("程序开始执行")

X = np.array([[0, 2], [1, 2], [2, 0]])

X = X.T

print("numpy的cov结果：", np.cov(X))

mean_X1 = 1
mean_X2 = 1

cov_X1X1 = np.sum((X[0, :] - mean_X1) ** 2) / (X.shape[1] - 1)
cov_X1X2 = np.sum((X[0, :] - mean_X1) * (X[1, :] - mean_X2)) / (X.shape[1] - 1)
cov_X2X1 = cov_X1X2
cov_X2X2 = np.sum((X[1, :] - mean_X2) ** 2) / (X.shape[1] - 1)

print("自己计算的结果：", np.array([[cov_X1X1, cov_X1X2], [cov_X2X1, cov_X2X2]]))


# # 修改代码增加异常处理
# import numpy as np
#
# print("程序开始执行")
#
# X = np.array([[0, 2], [1, 2], [2, 0]])
# X = X.T
#
# try:
#     print("X矩阵:", X)
#     result = np.cov(X)
#     print("numpy的cov结果：", result)
# except Exception as e:
#     print("numpy.cov执行出错:", e)
#
# mean_X1 = 1
# mean_X2 = 1
#
# try:
#     cov_X1X1 = np.sum((X[0, :] - mean_X1) ** 2) / (X.shape[1] - 1)
#     cov_X1X2 = np.sum((X[0, :] - mean_X1) * (X[1, :] - mean_X2)) / (X.shape[1] - 1)
#     cov_X2X1 = cov_X1X2
#     cov_X2X2 = np.sum((X[1, :] - mean_X2) ** 2) / (X.shape[1] - 1)
#
#     manual_result = np.array([[cov_X1X1, cov_X1X2], [cov_X2X1, cov_X2X2]])
#     print("自己计算的结果：", manual_result)
# except Exception as e:
#     print("手动计算出错:", e)
