import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import queue
import base64
import matplotlib.pyplot as plt
import time
from pid import PID
import hqyj_mqtt


# 新增：交通检测相关类和函数
class TrafficDetector:
    def __init__(self):
        # 加载YOLO模型
        self.model = YOLO('yolo11n.pt')  # 使用nano模型平衡速度和精度

        # 类别映射
        self.class_names = self.model.names
        self.traffic_light_classes = [9]  # YOLO中9是红绿灯
        self.person_class = 0  # YOLO中0是行人

        # 红绿灯状态判断参数
        self.red_threshold = 120  # 红色通道阈值
        self.green_threshold = 120  # 绿色通道阈值

    def detect_objects(self, frame):
        """检测图像中的物体并返回结果"""
        results = self.model(frame, verbose=False)
        return results[0]

    def get_traffic_light_state(self, roi):
        """判断红绿灯状态（基于颜色识别）"""
        if roi is None or roi.size == 0:
            return "unknown"

        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 红色范围
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        # 绿色范围
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # 计算颜色占比
        red_pixels = np.sum(red_mask > 0)
        green_pixels = np.sum(green_mask > 0)

        if red_pixels > green_pixels and red_pixels > 50:
            return "red"
        elif green_pixels > red_pixels and green_pixels > 50:
            return "green"
        else:
            return "unknown"

    def process_frame(self, frame):
        """处理单帧图像，返回检测结果"""
        results = self.detect_objects(frame)
        detection_info = {
            "traffic_light": "unknown",
            "has_person": False,
            "objects": []
        }

        # 绘制检测框
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

            # 检测行人
            if cls == self.person_class:
                detection_info["has_person"] = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 检测红绿灯并判断状态
            if cls in self.traffic_light_classes:
                # 提取红绿灯ROI
                roi = frame[y1:y2, x1:x2]
                light_state = self.get_traffic_light_state(roi)
                detection_info["traffic_light"] = light_state

                # 绘制红绿灯检测框及状态
                color = (0, 0, 255) if light_state == "red" else (0, 255, 0) if light_state == "green" else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Light: {light_state} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, detection_info


class LaneCenterPlotter:
    def __init__(self, max_frames=200, image_height=480):
        # 设置matplotlib为交互模式
        plt.ion()

        self.fig, self.ax = plt.subplots()
        self.line_lane_center, = self.ax.plot([], [], 'r-', label='Lane Center')
        self.line_image_center, = self.ax.plot([], [], 'b-', label='Image Center')

        # 设置图表、标题、和标签
        self.ax.set_title('Lane and Image Center')
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Pixel Coordinate')
        self.ax.legend()

        # 初始化数据
        self.x_data = []
        self.y_data_lane_center = []
        self.y_data_image_center = []
        self.max_frames = max_frames
        self.image_height = image_height

        # 初始化折线图
        self.init_plot()

    # 初始化折线图函数
    def init_plot(self):
        self.ax.set_xlim(0, self.max_frames)
        self.ax.set_ylim(0, self.image_height)
        self.line_lane_center.set_data([], [])
        self.line_image_center.set_data([], [])
        self.ax.grid()

    # 更新显示
    def update_plot(self, frame, lane_center, image_center):
        self.x_data.append(frame)
        self.y_data_lane_center.append(lane_center)
        self.y_data_image_center.append(image_center)

        # 更新折线图
        self.line_lane_center.set_data(self.x_data, self.y_data_lane_center)
        self.line_image_center.set_data(self.x_data, self.y_data_image_center)

        # 保持x轴的范围固定
        if len(self.x_data) > self.max_frames:
            self.ax.set_xlim(self.x_data[-self.max_frames], self.x_data[-1])
            self.ax.figure.canvas.draw()
        plt.pause(0.01)


def b64_to_np(image):
    # 此时的image还是一个字典，我们需要的是字典的值，字典的值是base64格式的图像数据
    img_data = base64.b64decode(image['image'])

    # 将字节数组转换numpy数组
    img_np = np.frombuffer(img_data, dtype=np.uint8)

    # 使用opencv读取该数组
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    return img


def perspective_transform(image):
    #              图像的 宽度          高度
    image_size = (image.shape[1], image.shape[0])

    # 调整透视变换的源点，使车道线检测更稳定
    src = np.float32(
        [[100, image_size[1]],
         [420, image_size[1]],
         [image_size[0] // 2 + 30, image_size[1] // 2 - 10],
         [image_size[0] // 2 - 30, image_size[1] // 2 - 10]]
    )

    dst = np.float32(
        [[image_size[0] / 4, image_size[1]],
         [image_size[0] * 3 / 4, image_size[1]],
         [image_size[0] * 3 / 4, 0],
         [image_size[0] / 4, 0]]
    )

    # 获取透视变换矩阵
    M = cv2.getPerspectiveTransform(src, dst)

    # 获取逆透视变换的矩阵
    minv = cv2.getPerspectiveTransform(dst, src)

    # 调用函数进行透视变换
    image_warp = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)

    return image_warp, minv


def dilate_erode(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image_dilate = cv2.dilate(image, kernel, iterations=1)
    result_img = cv2.erode(image_dilate, kernel, iterations=1)
    return result_img


def extract_line_gradient(image_warp):
    # 使用梯度的概念去提取车道线

    # 首先对传进来的鸟瞰图进行滤波
    img_Gaussian = cv2.GaussianBlur(image_warp, (5, 5), sigmaX=1)

    # 进行灰度化
    img_gray = cv2.cvtColor(img_Gaussian, cv2.COLOR_BGR2GRAY)

    # 使用sobel算子进行梯度的计算
    res = cv2.Sobel(img_gray, -1, 1, 0)

    # 做一个二值化，调整阈值使边缘检测更稳定
    ret, image_binary = cv2.threshold(res, 100, 255, cv2.THRESH_BINARY)

    res = dilate_erode(image_binary, 15)
    return res


# 提取白色车道线
def hlsSelect(img, thresh=(200, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    l_channel = l_channel / np.max(l_channel) * 255 if np.max(l_channel) > 0 else l_channel
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel < thresh[1])] = 1
    return binary_output


# 提取黄色车道线
def labSelect(img, thresh=(210, 225)):
    # 减少图像右侧的遮罩区域，提高检测准确性
    img[:, 300:, :] = (0, 0, 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab_b = lab[:, :, 2]
    if np.max(lab_b) > 100:
        lab_b = lab_b / np.max(lab_b) * 255
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b < thresh[1]))] = 1
    return binary_output


def extract_line_color(image_warp):
    # 提取白色车道线：HLS 模型
    hlsL_binary = hlsSelect(image_warp)

    # 提取黄色车道线：lab模型
    labB_binary = labSelect(image_warp)

    # 将提取到的白色车道线和黄色车道线进行融合
    combined_binary = np.zeros_like(hlsL_binary)
    combined_binary[(hlsL_binary == 1) | (labB_binary == 1)] = 1

    # 对融合后的车道线进行先膨胀后腐蚀的操作
    dilate_erode_image = dilate_erode(combined_binary, 15)
    return dilate_erode_image


def finding_line(dilate_erode_image):
    # 由于车是向前走的，在图像中表示为车是向上的，所以通常情况下，我们更关注黑白图像的下半部分
    histogram = np.sum(dilate_erode_image[dilate_erode_image.shape[0] // 2:, :], axis=0)

    # 创建一个三通道图像，用来显示小窗口寻找车道线的过程
    out_img = np.dstack((dilate_erode_image, dilate_erode_image, dilate_erode_image))

    # 获取直方图的中点位置，也就是图像宽度的一半
    midpoint = histogram.shape[0] // 2

    # 得到直方图左侧最高点的位置
    leftx_base = np.argmax(histogram[:midpoint]) if np.max(histogram[:midpoint]) > 100 else midpoint - 100

    # 直方图右侧最高点的位置
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint if np.max(histogram[midpoint:]) > 100 else midpoint + 100

    # 获取图像中所有非零像素的x和y的位置，返回的是行索引和列索引
    nonzero = dilate_erode_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # 定义一些小窗口的概念
    # 定义小窗口的个数
    nwindows = 10

    # 小窗口的高度
    window_height = dilate_erode_image.shape[0] // nwindows

    # 小窗口的宽度
    margin = 60  # 增加窗口宽度，提高检测稳定性

    # 小窗口内白色像素点的个数阈值
    minpix = 30  # 降低阈值，提高检测灵敏度

    # 初始化当前窗口的位置，后面会持续更新
    leftx_current = leftx_base
    rightx_current = rightx_base
    leftx_pre = leftx_current
    rightx_pre = rightx_current

    # 创建空列表接收左侧和右侧车道线像素的索引
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # 计算当前窗口的上边界的y坐标
        win_y_high = dilate_erode_image.shape[0] - (window + 1) * window_height

        # 计算当前窗口的下边界的y坐标
        win_y_low = dilate_erode_image.shape[0] - window * window_height

        # 计算左边窗口左右边界的x坐标
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        # 计算右边窗口的左右边界的x坐标
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # 找到处于窗口内非零像素的索引
        good_left_inds = ((nonzeroy >= win_y_high) & (nonzeroy < win_y_low) & (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_high) & (nonzeroy < win_y_low) & (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # 将获取到的白色像素点的索引添加到列表中
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 更新小窗口的位置
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        else:
            if len(good_right_inds) > minpix:
                offset = int(np.mean(nonzerox[good_right_inds])) - rightx_pre
                leftx_current = leftx_current + offset

        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
        else:
            if len(good_left_inds) > minpix:
                offset = int(np.mean(nonzerox[good_left_inds])) - leftx_pre
                rightx_current = rightx_current + offset

        # 记录上一次的位置
        leftx_pre = leftx_current
        rightx_pre = rightx_current

    # 连接索引的列表，为了后续更方便的提取出这些像素点的x和y的坐标，以便进行车道线的拟合
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 提取左侧和右侧车道线像素的位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 多项式拟合车道线，增加异常处理
    try:
        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 10 else np.array([0, 0, leftx_base])
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 10 else np.array([0, 0, rightx_base])
    except:
        left_fit = np.array([0, 0, leftx_base])
        right_fit = np.array([0, 0, rightx_base])

    # 生成用于绘制车道线的y坐标
    ploty = np.linspace(0, dilate_erode_image.shape[0] - 1, dilate_erode_image.shape[0])

    # 计算拟合出的车道线x坐标
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # 计算中间车道线的位置，增加平滑处理
    middle_fitx = (left_fitx + right_fitx) // 2

    # 对中间车道线进行平滑处理，减少抖动
    if len(middle_fitx) > 5:
        middle_fitx = np.convolve(middle_fitx, np.ones(5) / 5, mode='same')

    return left_fitx, right_fitx, middle_fitx, ploty


def show_line(image, image_warp, dilate_erode_image, minv, left_fitx, right_fitx, middle_fitx, ploty):
    # 创建空白图像绘制车道线
    warp_zero = np.zeros_like(dilate_erode_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # 组合车道线坐标
    pts_left = np.transpose(np.vstack([left_fitx, ploty]))
    pts_right = np.transpose(np.vstack([right_fitx, ploty]))
    pts_middle = np.transpose(np.vstack([middle_fitx, ploty]))

    # 绘制车道线
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(202, 124, 0), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(202, 124, 0), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_middle]), isClosed=False, color=(0, 255, 255), thickness=10)

    # 逆透视变换映射到原图
    newwarp = cv2.warpPerspective(color_warp, minv, (image.shape[1], image.shape[0]))

    # 融合结果
    result1 = cv2.addWeighted(image, 1, newwarp, 1, 0)

    # 创建灰色背景显示车道线
    background_zero = np.zeros_like(image).astype(np.uint8) + 127
    result = cv2.addWeighted(background_zero, 1, newwarp, 1, 0)

    # 拼接显示
    concatenate_image = np.concatenate((image, image_warp, result1, result), axis=1)
    cv2.imshow('concatenate_image', concatenate_image)
    return pts_middle


def process_detection(image, detector):
    # 调用TrafficDetector处理图像
    detected_frame, detection_info = detector.process_frame(image)

    # 显示检测结果
    cv2.imshow('Detection Result', detected_frame)

    # 仅返回检测状态，不直接设置速度
    detection_status = {
        "is_red_light": detection_info["traffic_light"] == "red",
        "has_person": detection_info["has_person"]
    }

    return detection_status, detected_frame


def auto_run(image, mqtt_client, pts_middle, pid, detection_status):
    # 自身基础速度设置，适当降低基础速度提高稳定性
    base_speed = 8  # 降低基础速度

    # 计算车道中心的像素坐标，使用更多点进行平均，减少抖动
    lane_center = pts_middle[180:, :].mean() if len(pts_middle) > 180 else image.shape[1] // 2
    image_center = image.shape[1] // 2

    # 计算偏差，增加死区，小偏差不调整方向
    error = lane_center - image_center
    if abs(error) < 10:  # 死区范围
        steering_angle = 0
    else:
        steering_angle = -pid(lane_center)

    # 从process_detection获取检测状态
    is_red_light = detection_status["is_red_light"]
    has_person = detection_status["has_person"]

    # 根据检测状态调整速度
    current_speed = base_speed
    if is_red_light:
        current_speed = 0  # 红灯停车
    elif has_person:
        current_speed = max(3, base_speed - 5)  # 有行人减速

    # 发送控制指令
    mqtt_client.send_mqtt(json.dumps({"carSpeed": current_speed}))
    mqtt_client.send_mqtt(json.dumps({"carDirection": steering_angle}))
    print(f'steering_angle: {steering_angle:.2f}, current_speed: {current_speed}, error: {error:.2f}')

    return lane_center, image_center


if __name__ == '__main__':
    q_mqtt_data = queue.Queue(5)

    plotter = LaneCenterPlotter()
    frame = 0

    detector = TrafficDetector()  # 实例化本地的TrafficDetector

    # 构建mqtt客户端
    mqtt_client = hqyj_mqtt.MQTTClient('127.0.0.1', 21883, 'bb', 'aa', q_mqtt_data)

    frame_count = 0
    start_time = time.time()

    # 调整PID参数，使控制更平稳
    pid = PID(Kp=0.25, Ki=0.01, Kd=0.2, setpoint=240)  # 降低Kp和Ki，增加Kd
    pid.sample_time = 0.1
    pid.output_limits = (-10, 10)  # 缩小转向角度范围，减少抖动

    while True:
        try:
            image = q_mqtt_data.get()

            if 'image' in image:
                # 转换图像格式
                image = b64_to_np(image)

                # 交通目标检测
                detection_status, detected_image = process_detection(image, detector)

                # 透视变换
                image_warp, minv = perspective_transform(image)

                # 提取车道线 - 尝试使用颜色提取方法，可能更稳定
                try:
                    # 先尝试使用颜色提取
                    dilate_erode_image = extract_line_color(image_warp)
                    # 如果提取效果不好，再使用梯度提取
                    if np.sum(dilate_erode_image) < 1000:
                        dilate_erode_image = extract_line_gradient(image_warp)
                except:
                    dilate_erode_image = extract_line_gradient(image_warp)

                # 拟合车道线
                left_fitx, right_fitx, middle_fitx, ploty = finding_line(dilate_erode_image)

                # 绘制车道线
                pts_middle = show_line(image, image_warp, dilate_erode_image, minv, left_fitx, right_fitx, middle_fitx,
                                       ploty)

                # 自动驾驶控制
                lane_center, image_center = auto_run(image, mqtt_client, pts_middle, pid, detection_status)

                # 更新图表
                plotter.update_plot(frame, lane_center, image_center)

                frame += 1

                # 按q退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error: {e}")
            # 发生错误时发送停车指令
            mqtt_client.send_mqtt(json.dumps({"carSpeed": 0}))
            mqtt_client.send_mqtt(json.dumps({"carDirection": 0}))

    cv2.destroyAllWindows()