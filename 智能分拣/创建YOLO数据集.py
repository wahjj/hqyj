import os
import random
import shutil

# 定义数据集目录和分割比例
source_root = 'dataset_t1'
target_root = 'dataset_y1'
train_ratio = 0.85
valid_ratio = 0.1
test_ratio = 0.05

# 创建目标文件夹及其子文件夹
train_dir = os.path.join(target_root, "train")
valid_dir = os.path.join(target_root, "val")
test_dir = os.path.join(target_root, "test")

for phase in ['train', 'test', 'val']:
    os.makedirs(os.path.join(target_root, phase, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_root, phase, 'labels'), exist_ok=True)

# 获取所有图像文件列表
images_dir = os.path.join(source_root, 'images')
labels_dir = os.path.join(source_root, 'labels')

# 确保源目录存在
if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
    raise FileNotFoundError(f"源目录不存在: images={images_dir}, labels={labels_dir}")

# 获取所有jpg文件名（不带扩展名）
image_files = [f.split('.')[0] for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]

# 随机打乱文件列表
random.shuffle(image_files)

# 计算分割点
num_files = len(image_files)
num_train = int(train_ratio * num_files)
num_valid = int(valid_ratio * num_files)

# 移动文件到目标位置
for i, filename in enumerate(image_files):
    # 构建源文件路径
    image_path = os.path.join(images_dir, f"{filename}.jpg")
    label_path = os.path.join(labels_dir, f"{filename}.txt")

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"警告: 图像文件不存在 - {image_path}")
        continue

    if not os.path.exists(label_path):
        print(f"警告: 标签文件不存在 - {label_path}")
        continue

    # 确定目标目录
    if i < num_train:
        dst_dir = train_dir
    elif i < num_train + num_valid:
        dst_dir = valid_dir
    else:
        dst_dir = test_dir

    # 复制文件
    try:
        shutil.copy(image_path, os.path.join(dst_dir, 'images'))
        shutil.copy(label_path, os.path.join(dst_dir, 'labels'))
    except Exception as e:
        print(f"复制文件时出错: {e}")

print("数据集划分完成！")