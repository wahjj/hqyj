import json
import paho.mqtt.client as mqtt
import time


class Mqtt_Clt():
    """
    用于表征Mqtt客户端的类
    """

    def __init__(self, ip_broker, port_broker, topic_sub, topic_pub, time_out_secs):
        """
        parameters:ip_broker:broker的IP地址
                   port_broker:连接服务的端口
                   topic_sub:订阅话题
                   topic_pub:发布话题
                   time_out_secs:连接超时的时间
        """
        self.mqtt_clt = mqtt.Client()  # 构造Mqtt客户端
        # self.mqtt_clt.on_connect = self.on_connect
        # self.mqtt_clt.on_connect_fail = self.on_connect_fail
        self.mqtt_clt.on_message = self.on_message  # 设置Mqtt客户端回调on_message为本类成员方法on_message
        self.topic_sub = topic_sub  # 设置订阅话题
        self.topic_pub = topic_pub  # 设置发布话题
        self.msg = {}  # 初始化接收消息的字典类成员变量msg为空({})
        self.mqtt_clt.connect(ip_broker, port_broker, time_out_secs)  # 连接Mqtt Broker
        self.mqtt_clt.subscribe(self.topic_sub, qos=0)  # 订阅相关话题
        self.mqtt_clt.loop_start()  # 开启接收循环

    def __del__(self):
        self.mqtt_clt.loop_forever()  # 结束接收循环
        self.mqtt_clt.disconnect()  # 断开连接

    def on_message(self, client, userdata, message):
        """
        parameters:client:回调返回的客户端实例
                   userdata:Client()或user_data_set()中设置的私有用户数据
                   msg:MQTTMessage实例,包含topic,payload,qos,retain
        functions:接收消息的回调,提取消息中的相关内容
        """
        pass
        # self.mqtt_queue.put(json.loads(message.payload.decode()))

    def send_json_msg(self, msg):
        """
        parameters:msg:json消息
        returns:无
        functions:在指定的话题上发布消息
        """
        self.mqtt_clt.publish(self.topic_pub, payload="{}".format(msg))

    def control_device(self, str_key, str_value):
        """
        parameters:str_key:键的字符串
                   str_value:值的字符串
        functions:控制设备
        """
        self.send_json_msg(json.dumps({str_key: str_value}))


if __name__ == '__main__':
    mqtt_client = Mqtt_Clt("127.0.0.1", 21883, "bb", "aa", 60)
    # 传送带运行  {"conveyor":"run"}
    # 推出1号杆  {"rod_control":"first_push"}
    # 拉回1号杆  {"rod_control":"first_pull"}
    mqtt_client.control_device("conveyor", 'run')
    while True:
        mqtt_client.control_device("rod_control", "first_push")
        time.sleep(1)
        mqtt_client.control_device("rod_control", "first_pull")
        time.sleep(1)
    # while True:
    #     mqtt_client.control_device("rod_control", "second_push")
    #     time.sleep(1)
    #     mqtt_client.control_device("rod_control", "second_pull")
    #     time.sleep(1)
    #     mqtt_client.control_device("rod_control", "third_push")
    #     time.sleep(1)
    #     mqtt_client.control_device("rod_control", "third_pull")
    #     time.sleep(1)
    #     mqtt_client.control_device("rod_control", "fourth_push")
    #     time.sleep(1)
    #     mqtt_client.control_device("rod_control", "fourth_pull")
    #     time.sleep(1)

