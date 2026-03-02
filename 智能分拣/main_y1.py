import json
import time
import threading

from hqyj_mqtt import Mqtt_Clt
import base64
import numpy as np
import cv2
from predict_y1 import predict_yolo

# 定义光推杆序号，以免后续编程时混淆
load_rod = "first"
ripe_rod = "second"
half_ripe_rod = "third"
raw_rod = "fourth"
all_rod = "all"
# 定义光电对管序号，以免后续编程时混淆
ripe_switch = {"first_switch": False}
half_ripe_switch = {"second_switch": False}
raw_switch = {"third_switch": False}


def push_pull(rod):
    mqtt_client.send_json_msg(json.dumps({"rod_control": f"{rod}_push"}))
    time.sleep(1)
    mqtt_client.send_json_msg(json.dumps({"rod_control": f"{rod}_pull"}))


if __name__ == '__main__':
    mqtt_client = Mqtt_Clt("127.0.0.1", 21883, "bb", "aa", 60)
    print("开始控制")
    mqtt_client.control_device("conveyor", 'run')

    mqtt_client.control_device("rod_control", "all_pull")
    threading.Timer(0.5, push_pull, args=(load_rod, )).start()
    result = None
    while True:
        msg = mqtt_client.mqtt_queue.get()
        # 如果获取的是图像数据
        if 'image' in msg:
            # 将 Base64 编码的字符串解码为原始二进制数据。
            image_data = base64.b64decode(msg['image'])
            # 将二进制数据转换为一个 np.uint8 类型的 NumPy 数组。
            image_array = np.frombuffer(image_data, np.uint8)
            # 将 NumPy 数组解码为 OpenCV 图像对象
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            # 识别
            result = predict_yolo(image)
            print(f"result: {result}")
        else:
            rod_control = None
            if (result == "ripe") and ripe_switch == msg:  # ripe 且 过第一个对管
                threading.Timer(0.0, push_pull, args=(ripe_rod,)).start()
                threading.Timer(0.2, push_pull, args=(load_rod,)).start()
            elif (result == "half-ripe") and half_ripe_switch == msg:  # half-ripe 且 过第二个对管
                threading.Timer(0.0, push_pull, args=(half_ripe_rod,)).start()
                threading.Timer(0.2, push_pull, args=(load_rod,)).start()
            elif (result == "raw") and raw_switch == msg:  # raw 且 过第三个对管
                threading.Timer(0.0, push_pull, args=(raw_rod,)).start()
                threading.Timer(0.2, push_pull, args=(load_rod,)).start()


