import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from pathlib import Path

# 数据集路径配置
RAW_DATA_FOLDER = './dataset/raw'
RIPE_DATA_FOLDER = './dataset/ripe'
HALF_DATA_FOLDER = './dataset/half-ripe'

# YOLOv8模型训练和预测相关路径
TRAINING_DATA_PATH = './training_dataset'
MODEL_SAVE_PATH = './models'
PREDICTION_OUTPUT_PATH = './predictions'


class YoloTrainerPredictor:
    def __init__(self):
        self.model = None
        self.class_names = ['unripe', 'half-ripe', 'ripe']  # 分别代表未成熟、半成熟、成熟

    def prepare_training_data(self):
        """
        准备训练数据，将不同成熟度的图片组织成YOLOv8所需的格式
        """
        # 创建目录结构
        train_images = os.path.join(TRAINING_DATA_PATH, 'images', 'train')
        val_images = os.path.join(TRAINING_DATA_PATH, 'images', 'val')
        train_labels = os.path.join(TRAINING_DATA_PATH, 'labels', 'train')
        val_labels = os.path.join(TRAINING_DATA_PATH, 'labels', 'val')

        os.makedirs(train_images, exist_ok=True)
        os.makedirs(val_images, exist_ok=True)
        os.makedirs(train_labels, exist_ok=True)
        os.makedirs(val_labels, exist_ok=True)

        # 复制图片到对应目录
        self._copy_and_label_images(RAW_DATA_FOLDER, train_images, train_labels, 0)  # 未成熟
        self._copy_and_label_images(HALF_DATA_FOLDER, train_images, train_labels, 1)  # 半成熟
        self._copy_and_label_images(RIPE_DATA_FOLDER, train_images, train_labels, 2)  # 成熟

        # 创建数据集配置文件
        self._create_dataset_yaml()

    def _copy_and_label_images(self, src_folder, img_dst_folder, lbl_dst_folder, class_id):
        """
        复制图片并创建对应的标签文件
        """
        if not os.path.exists(src_folder):
            print(f"警告: {src_folder} 不存在")
            return

        images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for i, img_name in enumerate(images):
            src_img_path = os.path.join(src_folder, img_name)

            # 为每个图片创建一个标签文件（这里假设每个图片只有一个目标，位于中心位置）
            # YOLO格式: class_id center_x center_y width height (归一化坐标)
            label_content = f"{class_id} 0.5 0.5 0.8 0.8\n"  # 假设目标在图片中心，占图片80%面积

            # 创建标签文件名
            base_name = os.path.splitext(img_name)[0]
            label_file_name = base_name + '.txt'

            # 复制图片
            dst_img_path = os.path.join(img_dst_folder, f"{base_name}_{class_id}_{i}.jpg")
            shutil.copy2(src_img_path, dst_img_path)

            # 写入标签文件
            label_file_path = os.path.join(lbl_dst_folder, f"{base_name}_{class_id}_{i}.txt")
            with open(label_file_path, 'w') as f:
                f.write(label_content)

    def _create_dataset_yaml(self):
        """
        创建YOLOv8所需的数据集配置文件
        """
        dataset_config = {
            'path': TRAINING_DATA_PATH,
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }

        config_path = os.path.join(TRAINING_DATA_PATH, 'dataset.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f)

    def train_model(self, epochs=50, img_size=640):
        """
        训练YOLOv8模型
        """
        # 加载预训练模型
        self.model = YOLO('yolov8n.pt')  # 使用YOLOv8 nano版本

        # 创建模型保存目录
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

        # 开始训练
        results = self.model.train(
            data=os.path.join(TRAINING_DATA_PATH, 'dataset.yaml'),
            epochs=epochs,
            imgsz=img_size,
            save_period=10,
            name='fruit_ripeness_detection'
        )

        # 保存最终模型
        final_model_path = os.path.join(MODEL_SAVE_PATH, 'best.pt')
        self.model.save(final_model_path)

        print(f"模型训练完成，已保存至: {final_model_path}")

    def load_model(self, model_path=None):
        """
        加载训练好的模型
        """
        if model_path is None:
            model_path = os.path.join(MODEL_SAVE_PATH, 'best.pt')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model = YOLO(model_path)
        print(f"模型加载成功: {model_path}")

    def predict_single_image(self, image_path, conf_threshold=0.5):
        """
        对单张图片进行预测
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model方法")

        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            save=False
        )

        result = results[0]
        annotated_img = result.plot()  # 绘制检测结果

        # 提取检测结果
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            detection = {
                'class_name': self.class_names[class_id],
                'confidence': confidence,
                'bbox': bbox
            }
            detections.append(detection)

        return detections, annotated_img

    def predict_from_raw_data(self, output_dir=None, conf_threshold=0.5):
        """
        从原始数据文件夹中读取图片并进行预测
        """
        if output_dir is None:
            output_dir = PREDICTION_OUTPUT_PATH

        os.makedirs(output_dir, exist_ok=True)

        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model方法")

        raw_images = [f for f in os.listdir(RAW_DATA_FOLDER)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        results_summary = {}

        for img_name in raw_images:
            img_path = os.path.join(RAW_DATA_FOLDER, img_name)

            try:
                detections, annotated_img = self.predict_single_image(img_path, conf_threshold)

                # 保存带注释的图片
                output_img_path = os.path.join(output_dir, f"annotated_{img_name}")
                cv2.imwrite(output_img_path, annotated_img)

                # 记录检测结果
                results_summary[img_name] = {
                    'detections': detections,
                    'output_path': output_img_path
                }

                print(f"处理完成: {img_name}, 检测到 {len(detections)} 个目标")

            except Exception as e:
                print(f"处理图片 {img_name} 时出错: {str(e)}")

        return results_summary

    def evaluate_model(self):
        """
        评估模型性能
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model方法")

        results = self.model.val()
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.p,
            'recall': results.box.r
        }

        print("模型评估结果:")
        print(f"mAP@0.5: {metrics['mAP50']:.4f}")
        print(f"mAP@0.5-0.95: {metrics['mAP50-95']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")

        return metrics


def main():
    """
    主函数 - 演示如何使用YoloTrainerPredictor类
    """
    trainer = YoloTrainerPredictor()

    # 步骤1: 准备训练数据
    print("正在准备训练数据...")
    trainer.prepare_training_data()
    print("训练数据准备完成!")

    # 步骤2: 训练模型 (可选)
    print("开始训练模型...")
    trainer.train_model(epochs=30)  # 可以调整训练轮数

    # 步骤3: 加载训练好的模型
    print("加载训练好的模型...")
    trainer.load_model()

    # 步骤4: 评估模型 (可选)
    print("评估模型性能...")
    trainer.evaluate_model()

    # 步骤5: 进行预测
    print("开始预测...")
    prediction_results = trainer.predict_from_raw_data()

    # 打印预测摘要
    print("\n预测摘要:")
    for img_name, result in prediction_results.items():
        detections = result['detections']
        print(f"{img_name}: 检测到 {len(detections)} 个目标")
        for det in detections:
            print(f"  - {det['class_name']} (置信度: {det['confidence']:.2f})")


if __name__ == "__main__":
    main()