import paho.mqtt.client as mqtt

broker = "127.0.0.1"
port = 21883
topic = "test/topic"
# 发布的消息内容
message = 'Hello, MQTT!'

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully.")
    else:
        print("Connection failed with code %d." % rc)

# 创建MQTT客户端实例
client = mqtt.Client()
# 设置连接回调函数
client.on_connect = on_connect
# 连接到MQTT代理
client.connect(broker, port, 60)
# 启动网络循环
client.loop_start()
# 发布消息
client.publish(topic, message)
client.loop_stop()
