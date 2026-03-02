import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1.读取数据集
dataset = pd.read_csv('dataset/iris.data')

# 添加列名
column_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
dataset.columns = column_names

# 通过head()函数，来检查数据的前几行，确保数据能够加载正确
print(dataset.head())

# 将前4列划分为特征X，最后一列class就是目标y
X = dataset.drop('class', axis=1)
y = dataset['class']

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)

# 使用标准化进行特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化朴素贝叶斯分类器
nb_classifier = GaussianNB()

# 根据划分好的特征去训练模型
nb_classifier.fit(X_train, y_train)


# 在测试集上进行预测
y_pred = nb_classifier.predict(X_test)

# 准确率：计算分类模型的准确率，也可以计算准确个数
accuracy = accuracy_score(y_test, y_pred)

# 混淆矩阵：可以直观的展示分类模型在各个类别上的预测情况。
# 返回一个二维数组（也就是混淆矩阵），矩阵的行表示真实类别，矩阵的列表示预测类别
conf_matrix = confusion_matrix(y_test, y_pred)#混淆矩阵

class_report = classification_report(y_test, y_pred)#分类报告

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n{conf_matrix}')
print(f'Classfication Report: \n{class_report}')

# 数据可视化，降维  使用t-sne算法进行降维操作，将特征讲到2维，方便可视化
tsne = TSNE(n_components=2)
x_tsne = tsne.fit_transform(X_test_scaled)
# 原始数据：4维特征（无法直接可视化）
# 降维后：2维特征（可以在平面图上展示）

# 将字符串标签转换为数值标签，方便画图
label_encoder = LabelEncoder()
y_test_numeric = label_encoder.fit_transform(y_pred)


# 绘制图像
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_test_numeric, cmap='viridis')
plt.title('t-SNE Visualization of naive_bayes Predictions')
plt.legend(*scatter.legend_elements(), title='Classes')
plt.show()