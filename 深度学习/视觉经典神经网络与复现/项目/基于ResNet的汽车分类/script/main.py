import numpy as np
import random
import torch
import os
import torch.nn as nn
import torchvision.models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ResNet import ResNet, BasicBlock
from sklearn.metrics import confusion_matrix
import seaborn as sns


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


setup_seed(0)


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU')


# 定义数据集的加载与处理
# 定义训练数据的处理步骤
train_tranforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 定义验证数据的处理步骤
valid_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 读取数据
train_dataset = datasets.ImageFolder('../dataset/train', transform=train_tranforms)
valid_dataset = datasets.ImageFolder('../dataset/val', transform=valid_transforms)


# 做成dataLoader，方便后续模型的训练
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True)

# 从训练集中抽几张图片进行显示
examples = enumerate(train_dataloader)
batch_idx, (imgs, lbs) = next(examples)
fig = plt.figure()

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(imgs[i][0], cmap='gray')
    plt.title(f'Ground Truth: {lbs[i]}')
    plt.xticks([])
    plt.yticks([])
plt.show()

# 实例化模型
# model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)

# 演示：使用预训练模型训练自己的任务
model = torchvision.models.resnet18(weights=None).to(device)

model.load_state_dict(torch.load('../model/resnet18-5c106cde.pth', weights_only=False))

for param in model.parameters():
    param.requires_grad = False

fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10),
    nn.LogSoftmax(dim=1),
).to(device)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 优化器

# 模型保存路径
save_path = '../model/last.pth'


# 训练模型
num_epoch = 5
for epoch in range(num_epoch):
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epoch}] Batch [{i + 1}/{len(train_dataloader)}] Loss {loss.item():.4f}')

    avg_loss = total_loss / len(train_dataloader)
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epoch}] Loss {avg_loss:.4f}')

torch.save(model.state_dict(), save_path)

# 模型评估
model.eval()

correct = 0
total = 0
predicted_labels = []
true_labels = []
with torch.no_grad():
    for images, labels in valid_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

print(f'Accuracy of the model on test images: {100 * correct / total:.2f}%')

# 可视化，绘制混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
