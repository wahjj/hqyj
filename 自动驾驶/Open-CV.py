import json

import hqyj_mqtt
import queue
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt

import time
from pid import PID


class LaneCenterPlotter:
    def __init__(self, max_frames=200, image_height=480):
        # 设置matplotlib为交互模式
        plt.ion()

        self.fig, self.ax = plt.subplots()
        self.line_lane_center,  = self.ax.plot([], [], 'r-', label='Lane Center')
        self.line_image_center,  = self.ax.plot([], [], 'b-', label='Image Center')

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


    # cv2.line(image, (80, image_size[1]), (image_size[0] // 2 - 60, image_size[1] // 2), (0, 0, 255), 1)
    # cv2.line(image, (450, image_size[1]), (image_size[0] // 2 + 60, image_size[1] // 2), (0, 0, 255), 1)


    src = np.float32(
        [[80, image_size[1]],
         [450, image_size[1]],
         [image_size[0] // 2 + 40, image_size[1] // 2 - 20],
         [image_size[0] // 2 - 40, image_size[1] // 2 - 20]]
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

    # 做一个二值化
    ret, image_binary = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)

    res = dilate_erode(image_binary, 15)
    # cv2.imshow('res', res)

    return res


# 提取白色车道线
def hlsSelect(img, thresh=(220, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    l_channel = l_channel / np.max(l_channel) * 255
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel < thresh[1])] = 1
    return binary_output


# 提取黄色车道线
def labSelect(img, thresh=(212, 220)):
    img[:, 240:, :] = (0, 0, 0)
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
    cv2.imshow('hlsL_binary', hlsL_binary)

    # 提取黄色车道线：lab模型
    labB_binary = labSelect(image_warp)
    cv2.imshow('labB_binary', labB_binary)

    # 将提取到的白色车道线和黄色车道线进行融合
    combined_binary = np.zeros_like(hlsL_binary)
    combined_binary[(hlsL_binary == 1) | (labB_binary == 1)] = 1

    # 对融合后的车道线进行先膨胀后腐蚀的操作‘
    dilate_erode_image = dilate_erode(combined_binary, 15)
    cv2.imshow('dilate_erode_image', dilate_erode_image)

    return dilate_erode_image


def finding_line(dilate_erode_image):
    # 由于车是向前走的，在图像中表示为车是向上的，所以通常情况下，我们更关注黑白图像的下半部分
    histogram = np.sum(dilate_erode_image[dilate_erode_image.shape[0] // 2:, :], axis=0)

    # 创建一个三通道图像，用来显示小窗口寻找车道线的过程
    out_img = np.dstack((dilate_erode_image, dilate_erode_image, dilate_erode_image))

    # 获取直方图的中点位置，也就是图像宽度的一半
    midpoint = histogram.shape[0] // 2

    # 得到直方图左侧最高点的位置
    leftx_base = np.argmax(histogram[:midpoint])

    # 直方图右侧最高点的位置
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

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
    margin = 50

    # 小窗口内白色像素点的个数阈值
    minpix = 40

    # 初始化当前窗口的位置，后面会持续更新
    leftx_current = leftx_base
    rightx_current = rightx_base
    leftx_pre = leftx_current
    # rightx_pre = leftx_current  这里应该是rightx_current，视频里写成了leftx_current
    # 但是最后运行没有出现异常的原因是，当照片里的两个车道线检测都是正常的时候，下面更新窗口时这两个变量不会去做运算，而是会被重新赋值
    # 所以在两条车道线检测都正常时，程序可以正常运行，但是这里需要改成rightx_current
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

        # 在out_img里去显示小窗口的滑动过程
        # cv2.rectangle(out_img, (win_xleft_low, win_y_high), (win_xleft_high, win_y_low), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_high), (win_xright_high, win_y_low), (0, 255, 0), 2)

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

    # print(left_lane_inds)
    # 提取左侧和右侧车道线像素的位置
    # left_lane_inds 是一个一维数组, 它包含了左侧车道线在滑动窗口中找到的白色像素点的x坐标的索引
    # 通过将这些索引作为索引器应用到 nonzerox数组上，就可以得到相应的左侧车道线的x坐标
    # leftx 包含了左侧车道线白色像素点的x坐标
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    # 有了坐标之后，就要去对左侧和右侧车道线进行多项式拟合，从而得到拟合的车道线
    # np.polyfit() 是numpy中用于进行多项式拟合的函数
    # 他接受三个参数：x y  和 deg
    # x：自变量数组，  y：因变量数组
    # deg：多项式的次数，如果是2   y = ax^2 + b^x + c
    # left_fit里存放的就是 a、b、c的参数，
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 使用np.linspace 生成一组均匀分布的数值，用于表示竖直方向上的像素坐标，方便后续的车道线的绘制
    ploty = np.linspace(0, dilate_erode_image.shape[0] - 1, dilate_erode_image.shape[0])

    # 使用多项式拟合来估计左侧和右侧车道线的x坐标
    # left_fitx 就是左侧拟合出来的车道线
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


    # 计算中间车道线的位置
    middle_fitx = (left_fitx + right_fitx) // 2
    #middle_fitx_smoothed = np.convolve(middle_fitx, np.ones(3) / 3, mode='same')#平滑

    # 使用不同颜色将车道线标出来
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    # cv2.imshow('out_img', out_img)

    return left_fitx, right_fitx, middle_fitx, ploty


def show_line(image, image_warp, dilate_erode_image, minv, left_fitx, right_fitx, middle_fitx, ploty):
    # 目标：显示原始图像，显示透视变换后的结果, 显示车道线在原图的结果，显示单纯车道线

    # 创建一个空白图像，用于在上面绘制检测到的车道线
    warp_zero = np.zeros_like(dilate_erode_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # 组合车道线坐标
    # print(left_fitx.shape)
    # print(right_fitx.shape)
    # print(ploty.shape)
    pts_left = np.transpose(np.vstack([left_fitx, ploty]))
    pts_right = np.transpose(np.vstack([right_fitx, ploty]))
    pts_middle = np.transpose(np.vstack([middle_fitx, ploty]))

    # print('pts_left.shape', pts_left.shape)
    # print('pts_right', pts_right.shape)
    # print('pts_middle', pts_middle.shape)

    # 绘制车道线
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(202, 124, 0), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(202, 124, 0), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_middle]), isClosed=False, color=(202, 124, 0), thickness=15)

    # 将得到的车道线的像素点根据逆透视变换映射到原始图像中
    newwarp = cv2.warpPerspective(color_warp, minv, (image.shape[1], image.shape[0]))

    # 将逆透视变换后的结果和原始图像进行融合
    result1 = cv2.addWeighted(image, 1, newwarp, 1, 0)


    # 创建一个灰色图
    background_zero = np.zeros_like(image).astype(np.uint8) + 127

    # 经过加权融合得到
    result = cv2.addWeighted(background_zero, 1, newwarp, 1, 0)

    # cv2.imshow('image_warp', image_warp)
    # cv2.imshow('result', result)
    # cv2.imshow('image', image)
    # cv2.imshow('result1', result1)

    concatenate_image = np.concatenate((image, image_warp, result1, result), axis=1)

    cv2.imshow('concatenate_image', concatenate_image)
    # cv2.waitKey(1)
    return pts_middle


def auto_run(image, mqtt_client, pts_middle, pid, carspeed=10):
    # 计算车道中心的像素坐标, 目标位置（其实是当前位置，因为在本案例中，目标位置和当前位置互换了）
    lane_center = pts_middle[240:, :].mean()
    # 当前位置（其实是目标位置）
    image_center = image.shape[1] // 2

    # 计算车道线弯曲程度（取中间车道线上半部分的x坐标标准差，值越大弯曲越剧烈）
    curve_degree = np.std(pts_middle[:100, 0])  # 取上半部分100个点的x坐标标准差

    # # 动态调整车速：弯曲越剧烈，车速越低
    # if curve_degree > 25:  # 阈值可根据实际场景调整
    #     carspeed = max(8, base_speed - 10)  # 连续急弯时降至10
    # elif curve_degree > 15:  # 中等弯曲
    #     carspeed = base_speed - 5  # 降至15
    # else:
    #     carspeed = base_speed  # 直线或缓弯保持原速

    steering_angle = -pid(lane_center)

    # 发送控制指令
    mqtt_client.send_mqtt(json.dumps({"carSpeed": carspeed}))
    mqtt_client.send_mqtt(json.dumps({"carDirection": steering_angle}))
    print('steering_angle', steering_angle)

    return lane_center, image_center

if __name__ == '__main__':
    q_mqtt_data = queue.Queue(10)

    plotter = LaneCenterPlotter()
    frame = 0

    # 1. 构建mqtt客户端，连接mqtt服务器，方便和3D场景通信
    mqtt_client = hqyj_mqtt.MQTTClient('127.0.0.1', 21883, 'bb', 'aa', q_mqtt_data)

    frame_count = 0
    start_time = time.time()

    pid = PID(Kp=0.315, Ki=0.0305, Kd=0.18, setpoint=240)
    pid.sample_time = 0.1
    pid.output_limits = (-13, 13)

    while True:
        try:
            image = q_mqtt_data.get()

            if 'image' in image:
                # frame_count += 1
                # current_time = time.time()
                # elapsed_time = current_time - start_time
                # fps = frame_count / elapsed_time if elapsed_time &gt; 0 else 0
                # print(f'fps:::::{fps:.2f}')


                # 将接收到的消息解析为opencv处理能用的格式
                image = b64_to_np(image)
                # image = cv2.imread('./2.png')

                # 对图像进行透视变换
                image_warp, minv = perspective_transform(image)
                # image_warp_copy = image_warp.copy()

                # 提取车道线：
                # 第一种 使用梯度来提取车道线
                dilate_erode_image = extract_line_gradient(image_warp)


                # 第二种 使用颜色来提取车道线
                # dilate_erode_image = extract_line_color(image_warp_copy)

                # 拟合的车道线
                left_fitx, right_fitx, middle_fitx, ploty = finding_line(dilate_erode_image)

                # 绘制车道线
                pts_middle = show_line(image, image_warp, dilate_erode_image, minv, left_fitx, right_fitx, middle_fitx, ploty)

                # 自动驾驶
                lane_center, image_center = auto_run(image, mqtt_client, pts_middle, pid)

                # 实时显示误差情况
                plotter.update_plot(frame, lane_center, image_center)
                frame += 1

        except Exception as e:
            print(e)