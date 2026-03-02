import torch
import numpy as np
from torch import nn

# 1.字符输入
text = "hey how are you"

# 3.数据集划分
input_seq = []
output_seq = []
window = 5
for i in range(0, len(text) - window, 1):
    input_seq.append(text[i:i + window])
    output_seq.append(text[i + window])
print("input_seq:", input_seq)
# print("output_seq:", output_seq)

# 4.数据编码：one-hot
chars = set(text)
chars = sorted(chars)
# print("chars:", chars)
# {" ":0, "a":1 }
char2int = {char: ind for ind, char in enumerate(chars)}
# print("char2int:", char2int)
# {0:" ", 1: "a"}
int2char = dict(enumerate(chars))

# 将字符转成数字编码
input_seq = [[char2int[char] for char in seq] for seq in input_seq]
# print("input_seq:", input_seq)
output_seq = [[char2int[char] for char in seq] for seq in output_seq]

# one-hot 编码
features = np.zeros((len(input_seq), len(chars)), dtype=np.float32)
for i, seq in enumerate(input_seq):
    features[i, seq] = 1.0
input_seq = torch.tensor(features, dtype=torch.float32)
features = np.zeros((len(output_seq), len(chars)), dtype=np.float32)
for i, seq in enumerate(output_seq):
    features[i, seq] = 1.0
output_seq = torch.tensor(features, dtype=torch.float32)

# 5.定义前向模型
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Model(len(chars), 32, len(chars))

# 6.定义损失函数和优化器
cri = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7.开始迭代
epochs = 1000
for epoch in range(1, epochs+1):
    output = model(input_seq)
    loss = cri(output, output_seq)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 8.显示频率设置
    if epoch == 0 or epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss {loss:.4f}")

# 预测下一个字符
input_text = "hey how a"
# 将字符转成数字编码
input_text = [char2int[char] for char in input_text]
print(input_text)
# one-hot 编码
features = np.zeros((len(chars)), dtype=np.float32)
print(features)
for seq in input_text:
    features[seq] = 1.0
input_text = torch.tensor(features, dtype=torch.float32)
out = model(input_text)
print("next char:", int2char[torch.argmax(out).item()])
