import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

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

# 使用标准化进行特征缩放（对KNN很重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化KNN分类器
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# 根据划分好的特征去训练模型（使用标准化后的数据）
knn_classifier.fit(X_train_scaled, y_train)

# 在测试集上进行预测（使用标准化后的数据）
y_pred = knn_classifier.predict(X_test_scaled)

# 准确率：计算分类模型的准确率，也可以计算准确个数
accuracy = accuracy_score(y_test, y_pred)

# 混淆矩阵：可以直观的展示分类模型在各个类别上的预测情况。
conf_matrix = confusion_matrix(y_test, y_pred)

class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n{conf_matrix}')
print(f'Classfication Report: \n{class_report}')

# 数据可视化，降维
tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(X_test_scaled)

# 将字符串标签转换为数值标签
label_encoder = LabelEncoder()
y_test_numeric = label_encoder.fit_transform(y_test)
unique_classes = label_encoder.classes_

# 绘制图像 - 方法1：为每个类别指定颜色
plt.figure(figsize=(10, 8))

# 定义颜色列表
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

for i, class_name in enumerate(unique_classes):
    mask = y_test_numeric == i
    plt.scatter(x_tsne[mask, 0], x_tsne[mask, 1],
                color=colors[i % len(colors)],  # 使用color而不是c和cmap
                label=class_name,
                s=50, alpha=0.7)

plt.title('t-SNE Visualization of KNN Predictions')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Classes')
plt.grid(True, alpha=0.3)
plt.show()