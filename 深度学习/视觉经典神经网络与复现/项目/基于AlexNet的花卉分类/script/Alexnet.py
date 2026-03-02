# _*_ coding: utf-8 _*_
# 导入所需的库
import os
import random

# 导入数据处理和可视化库
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 导入深度学习框架 PyTorch 相关库
import torch
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms


# 设置随机种子以保证结果的可重复性
def setup_seed(seed):
    np.random.seed(seed)  # 设置 Numpy 随机种子
    random.seed(seed)  # 设置 Python 内置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 哈希种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种子
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # 关闭 cudnn 加速
        torch.backends.cudnn.deterministic = True  # 设置 cudnn 为确定性算法


# 设置随机种子
setup_seed(0)

# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # 使用 CPU
    print("CUDA is not available. Using CPU.")

# 定义了一个名为transform的字典，其中包含两个键：“train”和“val”，分别对应于要应用于图像的特定转换集。
transform = {
    # 对训练集（train）的图像进行一系列数据增强和预处理操作
    "train": transforms.Compose([
        # 随机裁剪图像并调整大小为224x224像素
        # 这一步通常用于增加训练数据的多样性，防止模型过拟合
        transforms.RandomResizedCrop(224),

        # 以50%的概率随机水平翻转图像
        # 这是另一种常用的数据增强方法，有助于提高模型的鲁棒性
        transforms.RandomHorizontalFlip(p=0.5),

        # 将图像从PIL格式或NumPy数组格式转换为PyTorch张量
        # 张量是PyTorch中用于存储和操作数据的基本数据结构
        transforms.ToTensor(),

        # 对图像张量进行标准化处理
        # 参数 (0.5, 0.5, 0.5) 是每个通道的均值，(0.5, 0.5, 0.5) 是每个通道的标准差
        # 标准化公式为：output[channel] = (input[channel] - mean[channel]) / std[channel]
        # 这一步可以使数据分布更均匀，有助于模型更快收敛
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),

    # 对验证集（val）的图像进行一系列预处理操作
    "val": transforms.Compose([
        # 将图像调整为固定大小224x224像素
        # 验证集通常不需要数据增强，因此直接调整大小即可
        transforms.Resize((224, 224)),

        # 将图像转换为PyTorch张量
        transforms.ToTensor(),

        # 对图像张量进行标准化处理
        # 使用与训练集相同的均值和标准差，确保训练和验证数据的分布一致
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# 加载训练集和验证集的图像数据，并创建了对应的数据加载器。
train_dir = '../dataset/flower_data/train'
val_dir = '../dataset/flower_data/val'

# 使用 PyTorch 的 datasets.ImageFolder 加载训练集
# train_dir 是训练集图像所在的目录路径
# transform["train"] 指定了应用于训练集图像的变换操作（数据增强 + 预处理）
train_dataset = datasets.ImageFolder(
    train_dir,  # 训练集图像所在的根目录
    transform=transform["train"]  # 应用之前定义的训练集变换流水线
)

# 使用 PyTorch 的 datasets.ImageFolder 加载验证集
# val_dir 是验证集图像所在的目录路径
# transform["val"] 指定了应用于验证集图像的变换操作（仅预处理，无数据增强）
val_dataset = datasets.ImageFolder(
    val_dir,  # 验证集图像所在的根目录
    transform=transform["val"]  # 应用之前定义的验证集变换流水线
)

# # 定义一个列表来存储已经显示过的标签
# displayed_labels = []  # 用于记录已经显示过的类别标签，确保每个类别只显示一次
#
# # 遍历训练集，找到四张不同标签的图片
# count = 0  # 计数器，用于跟踪已经显示的图片数量
# fig, axes = plt.subplots(1, 4, figsize=(15, 3))  # 创建一个包含4个子图的画布，用于显示4张图片
# while count < 4:  # 循环直到找到并显示4张不同类别的图片
#     index = np.random.randint(len(train_dataset))  # 随机选择训练集中的一张图片索引
#     image, label = train_dataset[index]  # 根据索引获取图片及其对应的标签
#     if label not in displayed_labels:  # 如果该标签尚未被显示过
#         mean = np.array([0.5, 0.5, 0.5])  # 定义标准化时使用的均值
#         std = np.array([0.5, 0.5, 0.5])  # 定义标准化时使用的标准差
#
#         # 将标准化后的图像还原为原始像素值范围 [0, 1]
#         # 公式：image = image * std + mean
#         image = image.numpy() * std[:, None, None] + mean[:, None, None]
#
#         # 将图片的通道维度从 (C, H, W) 转换为 (H, W, C)，以便使用 matplotlib 显示
#         image = np.transpose(image, (1, 2, 0))
#
#         # 显示图片和标签
#         axes[count].imshow(image)  # 在当前子图中显示图片
#         axes[count].set_title(train_dataset.classes[label])  # 设置子图标题为图片的类别名称
#         axes[count].axis('off')  # 关闭坐标轴显示
#
#         displayed_labels.append(label)  # 将当前标签加入已显示标签列表
#         count += 1  # 增加计数器，表示已成功显示一张图片
# plt.show()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 定义模型
save_path = '../model'
if not os.path.exists(save_path):
    os.makedirs(save_path)

model = AlexNet(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:

        # 将数据移动到设备上
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), save_path + '/AlexNet.pth')


# 评估模型
correct = 0
total = 0
predicted_labels = []
true_labels = []

# 读取保存的模型
# model.load_state_dict(torch.load(save_path))
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

print(f'Accuracy of the model on the test images: {100 * correct / total:.4f} %')

# 生成混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
