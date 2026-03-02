import os
from ultralytics import YOLO


def validate_model():
    """验证训练后的模型"""
    model_path = 'runs/detect/train3/weights/best.pt'

    model = YOLO(model_path)

    # 在验证集上进行验证
    results = model.val(data='dataset_y1/data.yaml', imgsz=640, batch=16)

    # 显示验证结果
    print(f"🔍 验证结果:")
    print(f"   mAP@0.5: {results.box.map50:.4f}")
    print(f"   mAP@0.5-0.95: {results.box.map:.4f}")
    print(f"   Mean Precision: {results.box.mp:.4f}")
    print(f"   Mean Recall: {results.box.mr:.4f}")
    print(f"   F1-Score: {results.box.f1.mean():.4f}")

    # 显示各类别的详细结果
    print(f"\n📋 各类别详细指标:")
    names = ['raw', 'half-ripe', 'ripe']
    for i, name in enumerate(names):
        class_p, class_r, class_map50, class_map = results.box.class_result(i)
        print(f"   {name}: P={class_p:.4f}, R={class_r:.4f}, mAP50={class_map50:.4f}")


if __name__ == "__main__":
    # 确保数据集配置文件存在
    data_yaml_path = './dataset_y1/data.yaml'

    # 如果训练成功，进行验证
    res = validate_model()