import torch
import torch.nn.functional as F
from torchsummary import summary

# nn.relu  F.relu 其实是一样的
# F.relu一般用于函数调用，使用在forward中，而nn.relu是模块调用，使用在网络层定义中（__init__）
class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5, 1, 2)
        self.pool1 = torch.nn.AvgPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5, 1)
        self.pool2 = torch.nn.AvgPool2d(2, 2)
        self.fc1 = torch.nn.Linear(5 * 5 * 16, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = LeNet5().to("cuda" if torch.cuda.is_available() else "cpu")
    summary(model, (1, 28, 28))  # MNIST  28*28*1