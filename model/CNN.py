import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(CBL, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class CNNnet(nn.Module):
    def __init__(self, num_class):
        super(CNNnet, self).__init__()
        # 256*256*3 -> 128*128*64
        self.conv1 = CBL(3, 64, kernel_size=3, stride=1)
        # 128*128*64 -> 64*64*128
        self.conv2 = CBL(64, 128, kernel_size=3, stride=1)

        self.pool = nn.MaxPool2d(2, 2)
        # 第一层全连接(64 * 64 * 128 -> 500)
        self.fc1 = nn.Linear(64 * 64 * 128, 500)
        # 第二层全连接 (500 -> 10)
        self.fc2 = nn.Linear(500, num_class)
        # dropout层
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        # 将特征层展平
        x = self.dropout(x.reshape(-1, 64 * 64 * 128))
        x = self.dropout(F.relu(self.fc1(x)))
        out = self.fc2(x)
        return out
