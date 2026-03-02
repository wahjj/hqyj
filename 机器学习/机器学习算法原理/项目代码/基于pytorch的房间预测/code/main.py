import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


data = pd.read_excel('../dataset/Real estate valuation data set.xlsx')
print(data.head())

# 对便利店的数量做one-hot编码处理
# pd.get_dummies()函数会将指定列的每个不同取值转换为一个新的二进制列，新的列名由原列名和取值组成
data = pd.get_dummies(data, columns=['X4 number of convenience stores'])
print(data.keys())

X = data[['X1 transaction date', 'X2 house age',
       'X3 distance to the nearest MRT station', 'X5 latitude', 'X6 longitude',
       'X4 number of convenience stores_0',
       'X4 number of convenience stores_1',
       'X4 number of convenience stores_2',
       'X4 number of convenience stores_3',
       'X4 number of convenience stores_4',
       'X4 number of convenience stores_5',
       'X4 number of convenience stores_6',
       'X4 number of convenience stores_7',
       'X4 number of convenience stores_8',
       'X4 number of convenience stores_9',
       'X4 number of convenience stores_10']]
# print(X)

y = data['Y house price of unit area']

# 第一种方式：使用sklearn自带的函数去随机划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 第二种划分数据集的方式：将数据集按顺序划分训练集和测试集
# train_ratio = 0.8
# X, y = shuffle(X, y)
# X_train = X[:int(train_ratio * len(data))]
# X_test = X[int(train_ratio * len(data)):]
# y_train = y[:int(train_ratio * len(data))]
# y_test = y[int(train_ratio * len(data)):]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
           return self.linear(x)

model = LinearRegressionModel(X_train_tensor.shape[1])

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.1)

num_epochs = 1000
for epoch in range(num_epochs):
       model.train()

       optimizer.zero_grad()

       output = model(X_train_tensor)
       loss = criterion(output, y_train_tensor)

       loss.backward()

       optimizer.step()

       if (epoch+1) % 100 == 0:
              print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()

with torch.no_grad():
       predictions = model(X_test_tensor)

       test_loss = criterion(predictions, y_test_tensor)

       print(test_loss)


# 使用matplotlib进行绘制结果
predictions = predictions.numpy()
y_test_numpy = y_test_tensor.numpy()

# 创建第一个图形，用于绘制散点，展示预测值和实际值的对应关系
plt.figure(0)
plt.scatter(y_test_numpy, predictions, color='blue')
plt.plot([min(y_test_numpy), max(y_test_numpy)], [min(y_test_numpy), max(y_test_numpy)], linestyle='--', color='red',
         linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Regression results')

# 创建第二个图形
plt.figure(1)
sorted_indices = X_test.index.argsort()
y_test_sorted = y_test.iloc[sorted_indices]
y_pred_sorted = pd.Series(predictions.squeeze()).iloc[sorted_indices]

plt.plot(y_test_sorted.values, label='Acatual Values', marker='o')
plt.plot(y_pred_sorted.values, label='Predicted Values', marker='*')

# 设置轴标签和标题
plt.xlabel('Sorted Index')
plt.ylabel('Values')
plt.title('Actual vs Predicted Values in Linear Regression')

plt.show()