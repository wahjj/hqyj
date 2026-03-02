import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 假设数据保存在day.csv文件中，使用pandas的read_csv函数读取数据
# sep=','表示数据文件中的字段分隔符是逗号，读取后可通过查看数据头部确认读取是否正确
dataset = pd.read_csv('../dataset/day.csv', sep=',')

# 检查数据的前几行，确保数据加载正确
# 打印数据的前几行（默认是前5行），方便初步查看数据的结构、列名以及各列的数据类型等信息
print(dataset.head())

# 特征集，排除目标变量 'cnt', 'instant', 'dteday'
# 从原始数据集中选取要作为模型输入特征的列，组成特征矩阵X
# 'cnt' 可能是要预测的目标变量（例如可能代表某种数量），'instant' 和 'dteday' 可能是一些索引或者日期相关列，不适合作为特征参与建模
X = dataset.drop(['cnt', 'instant', 'dteday'], axis=1)
# 选取 'cnt' 列作为目标变量，即模型要预测的对象
y = dataset['cnt']


# 处理类别特征：使用独热编码
# 定义一个包含所有类别特征列名的列表，这些列通常包含离散的分类信息，如季节、年份等不同类别
categorical_features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
# 使用pd.get_dummies函数对这些类别特征列进行独热编码，将每个类别特征转换为多个二进制的虚拟变量列
# 例如，'season' 列如果有4个不同季节类别，会转换为4个新列，每个列对应一个季节，用0或1表示是否属于该季节
X_encoded = pd.get_dummies(X, columns=categorical_features)


# 划分训练集和测试集
# 这里注释掉了使用sklearn自带的随机划分函数train_test_split的方式
# 它可以按照指定的测试集比例（这里是0.2）和随机种子（random_state=42保证每次划分结果一致）来划分数据
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
# 将数据按顺序划分为训练集和测试集
# 根据给定的训练集占比（train_ratio = 0.8），按照数据的先后顺序进行划分
# 前80%的数据作为训练集，后20%的数据作为测试集
train_ratio = 0.8
X_train = X_encoded[:int(train_ratio * len(dataset))]
X_test = X_encoded[int(train_ratio * len(dataset)):]
y_train = y[:int(train_ratio * len(dataset))]
y_test = y[int(train_ratio * len(dataset)):]

# 使用标准化进行特征缩放
# 创建StandardScaler对象，用于对数据进行标准化处理，使数据的各特征维度具有均值为0，方差为1的分布特点
# 先在训练集上拟合标准化器（计算各特征的均值、方差等统计量），然后用拟合好的标准化器对训练集和测试集分别进行转换
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建线性回归模型
# 实例化sklearn中的线性回归模型类，这个模型会基于输入的特征和对应的目标变量学习一个线性关系，用于预测目标变量的值
linear_reg_model = LinearRegression()

# 训练模型
# 使用训练集数据（标准化后的特征和对应的目标变量）来训练线性回归模型，模型会自动拟合出最佳的系数（权重）
linear_reg_model.fit(X_train_scaled, y_train)

# 进行预测
# 使用训练好的线性回归模型对测试集特征数据（已经标准化）进行预测，得到预测值
y_pred = linear_reg_model.predict(X_test_scaled)

# 评估模型性能
# 计算均方误差（Mean Squared Error，MSE），它衡量了预测值与实际值之间的平均平方误差，值越小表示模型预测越准确
mse = mean_squared_error(y_test, y_pred)
# 计算R-squared值（决定系数），它表示模型对目标变量变异的解释程度，取值范围在0到1之间，越接近1表示模型拟合效果越好
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# 绘制预测结果和实际值的散点图
# 创建第一个图形，用于绘制散点图展示预测值和实际值的对应关系
plt.figure(0)
# 绘制散点图，横坐标为实际值，纵坐标为预测值，通过散点的分布可以直观查看模型预测的准确性和偏差情况
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values in Linear Regression')

# 按照时间顺序对数据排序
# 获取测试集索引的排序后的顺序，这里假设测试集索引对应着时间顺序或者其他有意义的顺序
sorted_indices = X_test.index.argsort()
# 根据排序后的索引获取对应的实际值
y_test_sorted = y_test.iloc[sorted_indices]
# 根据排序后的索引获取对应的预测值，并转换为pandas的Series类型
y_pred_sorted = pd.Series(y_pred).iloc[sorted_indices]

# 绘制实际值和预测值的曲线
# 创建第二个图形
plt.figure(1)
# 绘制实际值的曲线，用圆形标记，添加标签
plt.plot(y_test_sorted.values, label='Actual Values', marker='o')
# 绘制预测值的曲线，用叉号标记，添加标签
plt.plot(y_pred_sorted.values, label='Predicted Values', marker='x')
plt.xlabel('Sample Index (Sorted)')
plt.ylabel('Values')
plt.title('Actual vs Predicted Values in Linear Regression')
plt.legend()
# 显示绘制的图形，使图形窗口弹出展示结果
plt.show()