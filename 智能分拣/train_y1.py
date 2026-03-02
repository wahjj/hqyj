import os
from ultralytics import YOLO

def train_yolov8():
    """使用YOLOv8n.pt训练模型"""

    # 设置参数
    model_path = 'yolov8n.pt'  # 预训练模型路径
    dataset_path = './dataset_y1'  # 数据集根目录
    data_yaml = os.path.join(dataset_path, 'data.yaml')  # 数据集配置文件

    # 创建输出目录
    output_dir = 'runs/detect/train'
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)

    # 训练参数
    train_config = {
        'data': data_yaml,  # 数据集配置文件
        'epochs': 25,  # 训练轮数
        'imgsz': 640,  # 输入图像大小
        'batch': 16,  # 批次大小
    }

    # 开始训练
    results = model.train(**train_config)

    return results

def validate_model():
    """验证训练后的模型"""
    model_path = 'runs/detect/train3/weights/best.pt'

    model = YOLO(model_path)

    # 在验证集上进行验证
    results = model.val(data='dataset_y1/data.yaml', imgsz=640, batch=16)

    # 显示验证结果
    print(f"🔍 验证结果:")
    print(f"   mAP@0.5: {results.box.map:.4f}")
    print(f"   Precision: {results.box.precision:.4f}")
    print(f"   Recall: {results.box.recall:.4f}")



if __name__ == "__main__":
    # 确保数据集配置文件存在
    data_yaml_path = './dataset_y1/data.yaml'

    # 开始训练
    results = train_yolov8()

    # 如果训练成功，进行验证
    if results is not None:
        validate_model()