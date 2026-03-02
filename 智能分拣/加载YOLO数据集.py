import os
from PIL import Image, ImageDraw, ImageFont  # 导入PIL库中相关模块
import yaml

# 读取mydata.yaml配置
with open('./dataset/mydata.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

train_path = config['train']
val_path = config['val']
test_path = config['test']

# 类别名称映射
class_names = config['names']

# 数据集路径
img_path = os.path.join(train_path, "images")
label_path = os.path.join(train_path, "labels")
print(img_path)

# 处理前1张图像
for img_file in os.listdir(img_path)[:1]:
    if img_file.endswith('.jpg') or img_file.endswith('.png'):
        print(os.path.join(img_path, img_file))

        # 打开图像
        img = Image.open(os.path.join(img_path, img_file))
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # 打开对应的标签文件
        label_file = os.path.join(label_path, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

                # 计算边界框的坐标
                x_min = int((x_center - bbox_width / 2) * width)
                y_min = int((y_center - bbox_height / 2) * height)
                x_max = int((x_center + bbox_width / 2) * width)
                y_max = int((y_center + bbox_height / 2) * height)

                # 绘制边界框
                draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(0, 255, 0), width=2)

                # 绘制类别标签
                class_name = class_names[class_id]
                draw.text((x_min, y_min - 10), class_name, fill=(0, 255, 0))

        # 显示图像
        img.show()

