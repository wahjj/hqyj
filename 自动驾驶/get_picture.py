import hqyj_mqtt
import queue
import base64
import numpy as np
import cv2

def b64_to_np(image):
    # 此时的image还是一个字典，我们需要的是字典的值，字典的值是base64格式的图像数据
    img_data = base64.b64decode(image['image'])

    # 将字节数组转换numpy数组
    img_np = np.frombuffer(img_data, dtype=np.uint8)

    # 使用opencv读取该数组
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    return img


if __name__ == '__main__':

    q_mqtt_data = queue.Queue(5)
    i = 1

    # 1. 构建mqtt客户端，连接mqtt服务器，方便和3D场景通信
    mqtt_client = hqyj_mqtt.MQTTClient('127.0.0.1', 21883, 'bb', 'aa', q_mqtt_data)

    while True:
        image = q_mqtt_data.get()

        if 'image' in image:
            # 将接收到的消息解析为opencv处理能用的格式
            img = b64_to_np(image)
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            # print(key)
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'{i}.png', img)
                i += 1
                print('save successful')
            else:
                continue