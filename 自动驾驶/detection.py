import cv2
import numpy as np
import torch
from ultralytics import YOLO


class TrafficDetector:
    def __init__(self):
        # 加载YOLO模型
        self.model = YOLO('yolo11n.pt')  # 使用nano模型平衡速度和精度

        # 类别映射
        self.class_names = self.model.names
        self.traffic_light_classes = [9]  # YOLO中9是红绿灯
        self.person_class = 0  # YOLO中0是行人
        self.stop_line_class = 12  # 停止线类别（根据YOLO模型调整）

        # 红绿灯状态判断参数
        self.red_threshold = 120  # 红色通道阈值
        self.green_threshold = 120  # 绿色通道阈值

    def detect_objects(self, frame):
        """检测图像中的物体并返回结果"""
        results = self.model(frame, verbose=False)
        return results[0]

    def get_traffic_light_state(self, roi):
        """改进的红绿灯状态判断（基于颜色和亮度）"""
        if roi is None or roi.size == 0:
            return "unknown"

        # 确保ROI足够大
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            return "unknown"

        # 方法1: 基于亮度和颜色通道的方法
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 红色范围1 (0-10)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        # 红色范围2 (170-180)
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # 合并红色掩码
        red_mask = red_mask1 | red_mask2

        # 绿色范围
        lower_green = np.array([40, 80, 80])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # 黄色范围 (用于某些黄灯情况)
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 计算各颜色像素数量
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)

        # 方法2: 基于BGR通道的方法 (备选)
        b, g, r = cv2.split(roi)
        r_mean = np.mean(r)
        g_mean = np.mean(g)

        # 综合判断
        total_pixels = roi.shape[0] * roi.shape[1]

        # 设置最小像素阈值 (避免噪声干扰)
        min_pixel_threshold = max(10, total_pixels * 0.05)

        # 判断逻辑
        if red_pixels > min_pixel_threshold and red_pixels > green_pixels and red_pixels > yellow_pixels:
            return "red"
        elif green_pixels > min_pixel_threshold and green_pixels > red_pixels and green_pixels > yellow_pixels:
            return "green"
        elif yellow_pixels > min_pixel_threshold and yellow_pixels > red_pixels and yellow_pixels > green_pixels:
            return "yellow"  # 黄灯也视为需要注意的信号
        elif r_mean > g_mean + 30:  # 红色通道明显强于绿色
            return "red"
        elif g_mean > r_mean + 30:  # 绿色通道明显强于红色
            return "green"
        else:
            return "unknown"

    def process_frame(self, frame, warped_frame=None):
        """处理单帧图像，返回检测结果
        Args:
            frame: 原始图像（用于红绿灯、停止线和行人检测）
            warped_frame: 透视变换后的图像（保留参数但不使用）
        """
        # 检测原始图像中的物体
        results = self.detect_objects(frame)
        detection_info = {
            "traffic_light": "unknown",
            "has_person": False,
            "has_stop_line": False,
            "objects": []
        }

        # 限制行人检测区域：原图的中间区域（从左侧1/4到右侧1/4）
        height, width = frame.shape[:2]
        left_bound = width // 4
        right_bound = width * 3 // 4

        # 在原始图像上绘制检测区域边界（用于可视化）
        cv2.rectangle(frame, (left_bound, 0), (right_bound, height), (255, 0, 255), 2)
        cv2.putText(frame, "Person Detection Area", (left_bound, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 处理所有检测到的物体
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.5:  # 过滤低置信度检测
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.class_names[cls]

            # 记录检测到的物体
            detection_info["objects"].append({
                "label": label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

            # 检测停止线
            if cls == self.stop_line_class:
                detection_info["has_stop_line"] = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"Stop Line: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # 检测红绿灯并判断状态
            if cls in self.traffic_light_classes:
                # 提取红绿灯ROI (扩大一点范围以获得更好的颜色信息)
                expand = 5  # 扩大5个像素
                x1_exp = max(0, x1 - expand)
                y1_exp = max(0, y1 - expand)
                x2_exp = min(frame.shape[1], x2 + expand)
                y2_exp = min(frame.shape[0], y2 + expand)

                roi = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                light_state = self.get_traffic_light_state(roi)
                detection_info["traffic_light"] = light_state

                # 绘制红绿灯检测框及状态
                color = (0, 0, 255) if light_state == "red" else (0, 255, 0) if light_state == "green" else (
                0, 255, 255) if light_state == "yellow" else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Light: {light_state} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 检测行人（只在限定区域内）
            if cls == self.person_class:
                # 检查行人是否在限定区域内（计算边界框中心点）
                center_x = (x1 + x2) // 2
                if left_bound <= center_x <= right_bound:
                    detection_info["has_person"] = True
                    # 绘制行人检测框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # 在区域外的行人用不同颜色标记
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(frame, f"Person (outside): {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        return frame, detection_info