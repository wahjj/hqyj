# 导入库
import os
import random

import matplotlib.pyplot as plt
import numpy as np

# 导入torch的相关库
import torch
from torchvision import datasets, transforms

# 设置随机数种子，保证结果的可复现
def setup_seed(seed):
    np.random.seed(seed)  # Numpy的随机数种子
    random.seed(seed)  # Python的随机数种子
    os.environ["PYTHONHASHSEED"] = str(seed)  # 配置Python的哈希种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置cuda随机数种子
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # 关闭cuddn加速
        torch.backends.cudnn.deterministic = True  # 设置cudnn为确定性算法


# 设置随机数种子
setup_seed(0)

# 检查GPU、CUDA是否可用，如果可用就是用GPU，否则使用CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available, Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available, Using CPU.")


# 数据读取
train_dataset = datasets.MNIST(root="./dataset", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="./dataset", train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 显示6张图片
examples = enumerate(test_loader)
batch_idx, (imgs, labels) = next(examples)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(imgs[i][0], cmap="gray", interpolation="none")
    plt.title(f"Truth: {labels[i]}")
plt.show()
