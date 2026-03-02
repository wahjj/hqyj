import os
from hqyj_mqtt import Mqtt_Clt
import threading
import base64
import numpy as np
import cv2


RAW_DATA_FOLDER = './dataset/train/raw'
RIPE_DATA_FOLDER = './dataset/train/ripe'
HALF_DATA_FOLDER = './dataset/train/half-ripe'
TAKE_DATA_FOLDER = './dataset/take'


class GetData:
    def __init__(self, ip_broker, port, sub, pub, time_out, folder_path):
        self.mqtt_client = Mqtt_Clt(ip_broker, port, sub, pub, time_out)
        self.folder_path = folder_path
        self.make_dir(self.folder_path)
        self.t_recv_data = threading.Thread(target=self.recv_data)
        self.t_recv_data.start()

    def recv_data(self):
        i = 0
        while True:
            # 获取3D场景传输的数据
            image_base64 = self.mqtt_client.mqtt_queue.get()
            # 如果获取的是图像数据
            if 'image' in image_base64:
                # 将 Base64 编码的字符串解码为原始二进制数据。
                print(image_base64)
                image_data = base64.b64decode(image_base64['image'])
                # 将二进制数据转换为一个 np.uint8 类型的 NumPy 数组。
                image_array = np.frombuffer(image_data, np.uint8)
                # 将 NumPy 数组解码为 OpenCV 图像对象
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                cv2.imwrite(self.folder_path + '/' + f'img{i}.jpg', image)
                print(f'图片{i}已保存')
                i += 1

    def make_dir(self, folder_path):
        if not os.path.exists(folder_path):
            print(f'{folder_path}不存在')
            os.makedirs(folder_path)
            print(f'{folder_path}已创建')
        else:
            print(f'{folder_path}已存在，无需创建')


get_data = GetData('127.0.0.1', 21883, 'bb', 'aa', 60, TAKE_DATA_FOLDER)
