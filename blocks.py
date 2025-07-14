
import torch
import torch.nn as nn
import torch.nn.functional as F

# 基础卷积模块
class ConvBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# 边缘增强模块
class ELA(nn.Module):
    def __init__(self, channels=64):
        super(ELA, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        edge = self.conv1(x)
        out = x + self.conv2(edge)
        return out

# MambaBlock（简化）
class MambaBlock(nn.Module):
    def __init__(self, dim=64):
        super(MambaBlock, self).__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)

# BAM 注意力模块
class BAM(nn.Module):
    def __init__(self, channels=64):
        super(BAM, self).__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att(x)

# AdaIN 模块
class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, x, y, eps=1e-5):
        b, c, h, w = x.size()
        x_mean = x.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        x_std = x.view(b, c, -1).std(dim=2).view(b, c, 1, 1) + eps
        y_mean = y.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        y_std = y.view(b, c, -1).std(dim=2).view(b, c, 1, 1) + eps
        return (x - x_mean) / x_std * y_std + y_mean

# DenSoA 注意力模块
class DenSoA(nn.Module):
    def __init__(self, channels=64):
        super(DenSoA, self).__init__()
        self.att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att(x)
