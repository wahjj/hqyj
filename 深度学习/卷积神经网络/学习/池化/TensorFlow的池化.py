import tensorflow as tf

# 输入特征张量
# [batch_size, input_height, input_width, in_channels]
# 这里创建了一个形状为 (1, 5, 5, 1) 的输入张量，代表一个批次，大小为5x5的图像，1个通道
input = tf.constant([[
[[1], [0], [0], [0], [1]],
[[0], [1], [0], [1], [0]],
[[0], [0], [1], [0], [0]],
[[0], [1], [0], [1], [0]],
[[1], [0], [0], [0], [1]]
]], dtype=tf.float32)

# 卷积核张量
# [kernel_height, kernel_width, in_channels, out_channels]
# 这里创建了一个形状为 (3, 3, 1, 1) 的卷积核，代表3x3的滤波器，1个输入通道，1个输出通道
wc1 = tf.constant([[
[[0]], [[0]], [[1]]],
[[[0]], [[1]], [[0]]],
[[[1]], [[0]], [[0]]]
], dtype=tf.float32)

# 创建卷积层
# 使用 tf.nn.conv2d 进行卷积操作，strides=[1, 1, 1, 1] 表示在每个维度上移动1步，padding='SAME' 填充使输出与输入相同大小
conv_layer1 = tf.nn.conv2d(input, wc1, strides=[1, 1, 1, 1], padding='SAME')

# 将卷积的输出转为 NumPy 数组
output1 = conv_layer1.numpy()

# 获取卷积层输出特征图的形状
batch_size = output1.shape[0] # 批次大小
height = output1.shape[1] # 高度
width = output1.shape[2] # 宽度
channels = output1.shape[3] # 通道数

# 打印卷积后特征图的形状和值
print(f"卷积后的特征大小: [batch_size={batch_size}, height={height}, width={width}, channels={channels}]")
print("输出特征图:", output1.reshape(height, width)) # 以 (height, width) 的形状打印输出特征图



# 添加池化操作
# 在卷积后的特征图上进行平均池化操作，ksize=[1, 2, 2, 1] 表示在 2x2 的区域内进行池化，strides=[1, 1, 1, 1] 表示步幅为 1，padding='SAME' 表示填充
pool_layer1 = tf.nn.avg_pool(conv_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 获取池化后的输出特征图并将其转为 NumPy 数组
output_pool_layer1 = pool_layer1.numpy()

# 获取池化后输出特征图的形状
pool_batch_size = output_pool_layer1.shape[0] # 批次大小
pool_height = output_pool_layer1.shape[1] # 高度
pool_width = output_pool_layer1.shape[2] # 宽度
pool_channels = output_pool_layer1.shape[3] # 通道数

# 打印池化后的特征图的形状和值
print(f"池化后的特征大小: [batch_size={pool_batch_size}, height={pool_height}, width={pool_width}, channels={pool_channels}]")
print("输出特征图:", output_pool_layer1.reshape(pool_height, pool_width)) # 将池化后的输出特征图调整为 (pool_height, pool_width) 的形状并打印
