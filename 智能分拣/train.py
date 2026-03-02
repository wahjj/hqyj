import os
import shutil
from ultralytics import YOLO


# =============================
# 1. 数据集准备和验证
# =============================
def check_dataset_structure(dataset_root):
    """检查数据集目录结构是否正确"""
    required_dirs = ['train', 'val', 'test']
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_root, dir_name)
        if not os.path.exists(dir_path):
            print(f"警告: 目录 {dir_path} 不存在")
            return False

    # 检查每个目录下的images和labels子目录
    for phase in ['train', 'val', 'test']:
        images_dir = os.path.join(dataset_root, phase, 'images')
        labels_dir = os.path.join(dataset_root, phase, 'labels')

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"警告: {phase} 目录缺少 images 或 labels 子目录")
            return False

        # 检查是否有文件
        if len(os.listdir(images_dir)) == 0:
            print(f"警告: {phase}/images 目录为空")
            return False

    print("✅ 数据集结构检查通过")
    return True


# =============================
# 2. 训练配置
# =============================
def train_yolov8():
    """使用YOLOv8n.pt训练模型"""

    # 设置参数
    model_path = 'yolov8n.pt'  # 预训练模型路径
    dataset_path = './dataset_y1'  # 数据集根目录
    data_yaml = os.path.join(dataset_path, 'data.yaml')  # 数据集配置文件

    # 检查数据集结构
    if not check_dataset_structure(dataset_path):
        print("❌ 数据集结构不完整，请检查后重试")
        return

    # 创建输出目录
    output_dir = 'runs/detect/train'
    os.makedirs(output_dir, exist_ok=True)

    # 加载预训练模型
    print("🚀 加载预训练模型...")
    model = YOLO(model_path)

    # 训练参数
    train_config = {
        'data': data_yaml,  # 数据集配置文件
        'epochs': 50,  # 训练轮数
        'imgsz': 640,  # 输入图像大小
        'batch': 16,  # 批次大小
        # 'name': 'custom_train',  # 运行名称
        # 'device': 'cpu' ,  # 设备  if torch.cuda.is_available() else 'cpu'
        # 'workers': 4,  # 工作线程数
        # 'lr0': 0.01,  # 初始学习率
        # 'optimizer': 'SGD',  # 优化器
        # 'weight_decay': 0.0005,
        # 'patience': 10,  # 早停耐心值
        # 'save_period': 10,  # 每多少个epoch保存一次
        # 'verbose': True,  # 显示详细信息
        # 'project': 'runs/detect'  # 项目目录
    }

    # 开始训练
    print("🎯 开始训练...")
    try:
        results = model.train(**train_config)
        print("✅ 训练完成！")

        # 保存最佳模型
        best_model_path = os.path.join(output_dir, 'model/best.pt')
        print(f"🎉 最佳模型已保存到: {best_model_path}")

        return results

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        return None


# =============================
# 3. 验证训练结果
# =============================
def validate_model():
    """验证训练后的模型"""
    model_path = 'runs/detect/train/weights/best.pt'
    if not os.path.exists(model_path):
        print("⚠️ 未找到训练好的模型，跳过验证")
        return

    model = YOLO(model_path)

    # 在验证集上进行验证
    results = model.val(data='dataset_y1/data.yaml', imgsz=640, batch=16)

    # 显示验证结果
    print(f"🔍 验证结果:")
    print(f"   mAP@0.5: {results.box.map:.4f}")
    print(f"   Precision: {results.box.precision:.4f}")
    print(f"   Recall: {results.box.recall:.4f}")



if __name__ == "__main__":
    # 安装依赖（如果需要）
    # pip install ultralytics

    # 确保数据集配置文件存在
    data_yaml_path = './dataset_y1/data.yaml'
    if not os.path.exists(data_yaml_path):
        print("❌ 错误: data.yaml 文件不存在")
        print("请确保数据集目录中包含正确的data.yaml文件")
        exit(1)

    # 开始训练
    results = train_yolov8()

    # 如果训练成功，进行验证
    if results is not None:
        validate_model()