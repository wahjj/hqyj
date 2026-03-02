# _*_ coding: utf-8 _*_
# 导入所需的库
import os
import random


# 导入数据处理和可视化库
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 导入深度学习框架 PyTorch 相关库
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torchvision import datasets, transforms
from ResNet import ResNet, BasicBlock

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
setup_seed(5)
# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # 使用 CPU
    print("CUDA is not available. Using CPU.")

train_transforms = transforms.Compose(
    [transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),  # 随机水平翻转
     transforms.ToTensor(),  # 转化成张量
     transforms.Normalize([0.485, 0.456, 0.406],  # 归一化
                          [0.229, 0.224, 0.225])
     ])

valid_transforms = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])])

# # 读取数据
train_dataset = datasets.ImageFolder('../dataset/train', transform=train_transforms)
test_dataset = datasets.ImageFolder('../dataset/val', transform=valid_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 显示6张图片

# 读取数据,batch_idx从0开始
examples = enumerate(train_loader)
batch_idx, (imgs, labels) = next(examples)
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(imgs[i][0])
    plt.title("Ground Truth: {}".format(labels[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()

model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2).to(device)
save_path = '../model/last.pth'

#
# model = torchvision.models.resnet18(pretrained=True).to(device)
# model.load_state_dict(torch.load('../model/resnet18-5c106cde.pth'))
# for param in model.parameters():
#     param.requires_grad = False
#
# fc_inputs = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Linear(fc_inputs, 256),
#     nn.ReLU(),
#     nn.Dropout(0.4),
#     nn.Linear(256, 2),
#     nn.LogSoftmax(dim=1)
# ).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
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
        # 打印每个batch的损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), save_path)

# 评估模型
correct = 0
total = 0
predicted_labels = []
true_labels = []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# 生成混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
