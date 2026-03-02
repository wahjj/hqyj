# 导入库
import os
import random

import numpy as np

# 导入torch的相关库
import torch

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