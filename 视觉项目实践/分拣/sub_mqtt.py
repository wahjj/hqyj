import paho.mqtt.client as mqtt

broker = "127.0.0.1"
port = 21883
topic = "test/topic"

# 当连接到MQTT代理时调用
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully.")
        # 订阅主题
        client.subscribe(topic)
    else:
        print("Connection failed with code %d." % rc)

# 当接收到消息时调用
def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload.decode()} on topic {msg.topic}")

# 创建MQTT客户端实例
client = mqtt.Client()
# 设置连接和消息回调函数
client.on_connect = on_connect
client.on_message = on_message
# 连接到MQTT代理
client.connect(broker, port)
# 启动网络循环
client.loop_forever()
