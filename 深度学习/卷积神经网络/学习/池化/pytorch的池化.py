import torch
import torch.nn as nn

# 1. 定义输入数据 (Python 列表)
# 格式看起来是 (Batch=1, Height=5, Width=5, Channels=1)
input_data_list = [[
  [[1], [0], [1], [0], [2]],
  [[0], [1], [0], [3], [1]],
  [[1], [0], [3], [0], [1]],
  [[0], [3], [0], [1], [0]],
  [[2], [0], [1], [0], [1]]
]]

# 2. 将列表转换为 PyTorch 张量，并指定为浮点类型
# 初始张量形状: [1, 5, 5, 1] (N, H, W, C)
input_tensor_nhwc = torch.tensor(input_data_list, dtype=torch.float32)
print(f"初始张量形状 (NHWC): {input_tensor_nhwc.shape}")
print("初始张量内容 (NHWC):\n", input_tensor_nhwc)

# 3. 调整维度顺序以匹配 PyTorch 的要求 (N, C, H, W)
# 使用 permute(0, 3, 1, 2) 将 N H W C 变为 N C H W
input_tensor_nchw = input_tensor_nhwc.permute(0, 3, 1, 2)
print(f"\n调整后张量形状 (NCHW): {input_tensor_nchw.shape}")
print("调整后张量内容 (NCHW):\n", input_tensor_nchw)

# 4. 定义池化层
kernel_size = 2  # 池化核大小
stride = 2       # 池化步长

# 使用最大池化 Max Pooling
# pool_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
# 如果想用平均池化，可以使用:
pool_layer = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

print(f"\n使用的池化层: {pool_layer}")

# 5. 应用池化操作
output_tensor = pool_layer(input_tensor_nchw)

# 6. 打印输出结果
print(f"\n池化后张量形状: {output_tensor.shape}")
print("池化后张量内容:\n", output_tensor)
