# -*- coding: utf-8 -*-
from torch import nn


#输入图像 112 × 112
class Conv2fc1_avg_16(nn.Module):
    def __init__(self):
        super(Conv2fc1_avg_16, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(16))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.AvgPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(32))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.AvgPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(32*28*28, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv2fc1_avg_32(nn.Module):
    def __init__(self):
        super(Conv2fc1_avg_32, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.AvgPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.AvgPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 28 * 28, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv2fc1_avg_64(nn.Module):
    def __init__(self):
        super(Conv2fc1_avg_64, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.AvgPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.AvgPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(128 * 28 * 28, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv2fc1_max_16(nn.Module):
    def __init__(self):
        super(Conv2fc1_max_16, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(16))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.MaxPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(32))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.MaxPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(32 * 28 * 28, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv2fc1_max_32(nn.Module):
    def __init__(self):
        super(Conv2fc1_max_32, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.MaxPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.MaxPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 28 * 28, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv2fc1_max_64(nn.Module):
    def __init__(self):
        super(Conv2fc1_max_64, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.MaxPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.MaxPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(128 * 28 * 28, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv2fc1_nopool_16(nn.Module):
    def __init__(self):
        super(Conv2fc1_nopool_16, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(16))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(32))
        layer1.add_module('relu12', nn.ReLU(True))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(32 * 28 * 28, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv2fc1_nopool_32(nn.Module):
    def __init__(self):
        super(Conv2fc1_nopool_32, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 28 * 28, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv2fc1_nopool_64(nn.Module):
    def __init__(self):
        super(Conv2fc1_nopool_64, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(128 * 28 * 28, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc1_max_16(nn.Module):
    def __init__(self):
        super(Conv3fc1_max_16, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(16))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.MaxPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(32))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.MaxPool2d(2))
        layer1.add_module('conv13', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(64))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.MaxPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 14 * 14, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc1_max_32(nn.Module):
    def __init__(self):
        super(Conv3fc1_max_32, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.MaxPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.MaxPool2d(2))
        layer1.add_module('conv13', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(128))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.MaxPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(128 * 14 * 14, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc1_max_64(nn.Module):
    def __init__(self):
        super(Conv3fc1_max_64, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.MaxPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.MaxPool2d(2))
        layer1.add_module('conv13', nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(256))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.MaxPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(256 * 14 * 14, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc1_avg_16(nn.Module):
    def __init__(self):
        super(Conv3fc1_avg_16, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(16))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.AvgPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(32))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.AvgPool2d(2))
        layer1.add_module('conv13', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(64))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.AvgPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 14 * 14, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc1_avg_32(nn.Module):
    def __init__(self):
        super(Conv3fc1_avg_32, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.AvgPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.AvgPool2d(2))
        layer1.add_module('conv13', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(128))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.AvgPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(128 * 14 * 14, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc1_avg_64(nn.Module):
    def __init__(self):
        super(Conv3fc1_avg_64, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.AvgPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.AvgPool2d(2))
        layer1.add_module('conv13', nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(256))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.AvgPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(256 * 14 * 14, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc1_nopool_16(nn.Module):
    def __init__(self):
        super(Conv3fc1_nopool_16, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(16))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(32))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(64))
        layer1.add_module('relu13', nn.ReLU(True))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 14 * 14, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc1_nopool_32(nn.Module):
    def __init__(self):
        super(Conv3fc1_nopool_32, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(128))
        layer1.add_module('relu13', nn.ReLU(True))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(128 * 14 * 14, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


# 输入图像 112 × 112
class Conv3fc1_nopool_64(nn.Module):
    def __init__(self):
        super(Conv3fc1_nopool_64, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(256))
        layer1.add_module('relu13', nn.ReLU(True))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(256 * 14 * 14, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc1_nopool_avg_16(nn.Module):
    def __init__(self):
        super(Conv3fc1_nopool_avg_16, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(16))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(32))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(64))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.AvgPool2d(14))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc1_nopool_avg_32(nn.Module):
    def __init__(self):
        super(Conv3fc1_nopool_avg_32, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(128))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.AvgPool2d(14))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(128, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc1_nopool_avg_64(nn.Module):
    def __init__(self):
        super(Conv3fc1_nopool_avg_64, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(256))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.AvgPool2d(14))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(256, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc2_max_16(nn.Module):
    def __init__(self):
        super(Conv3fc2_max_16, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(16))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.MaxPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(32))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.MaxPool2d(2))
        layer1.add_module('conv13', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(64))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.MaxPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 14 * 14, 512))
        layer3.add_module('fc2', nn.Linear(512, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc2_max_32(nn.Module):
    def __init__(self):
        super(Conv3fc2_max_32, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.MaxPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.MaxPool2d(2))
        layer1.add_module('conv13', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(128))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.MaxPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(128 * 14 * 14, 512))
        layer3.add_module('fc2', nn.Linear(512, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc2_max_64(nn.Module):
    def __init__(self):
        super(Conv3fc2_max_64, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.MaxPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.MaxPool2d(2))
        layer1.add_module('conv13', nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(256))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.MaxPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(256 * 14 * 14, 512))
        layer3.add_module('fc2', nn.Linear(512, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc2_avg_16(nn.Module):
    def __init__(self):
        super(Conv3fc2_avg_16, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(16))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.AvgPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(32))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.AvgPool2d(2))
        layer1.add_module('conv13', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(64))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.AvgPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 14 * 14, 512))
        layer3.add_module('fc2', nn.Linear(512, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc2_avg_32(nn.Module):
    def __init__(self):
        super(Conv3fc2_avg_32, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.AvgPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.AvgPool2d(2))
        layer1.add_module('conv13', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(128))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.AvgPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(128 * 14 * 14, 512))
        layer3.add_module('fc2', nn.Linear(512, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc2_avg_64(nn.Module):
    def __init__(self):
        super(Conv3fc2_avg_64, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('pool11', nn.AvgPool2d(2))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool12', nn.AvgPool2d(2))
        layer1.add_module('conv13', nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(256))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.AvgPool2d(2))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(256 * 14 * 14, 512))
        layer3.add_module('fc2', nn.Linear(512, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc2_nopool_16(nn.Module):
    def __init__(self):
        super(Conv3fc2_nopool_16, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(16))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(32))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(64))
        layer1.add_module('relu13', nn.ReLU(True))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 14 * 14, 512))
        layer3.add_module('fc2', nn.Linear(512, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc2_nopool_32(nn.Module):
    def __init__(self):
        super(Conv3fc2_nopool_32, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(128))
        layer1.add_module('relu13', nn.ReLU(True))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(128 * 14 * 14, 512))
        layer3.add_module('fc2', nn.Linear(512, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc2_nopool_64(nn.Module):
    def __init__(self):
        super(Conv3fc2_nopool_64, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(256))
        layer1.add_module('relu13', nn.ReLU(True))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(256 * 14 * 14, 512))
        layer3.add_module('fc2', nn.Linear(512, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc2_nopool_avg_16(nn.Module):
    def __init__(self):
        super(Conv3fc2_nopool_avg_16, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(16))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(32))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(64))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.AvgPool2d(14))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 , 512))
        layer3.add_module('fc2', nn.Linear(512, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc2_nopool_avg_32(nn.Module):
    def __init__(self):
        super(Conv3fc2_nopool_avg_32, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(128))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.AvgPool2d(14))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(128, 512))
        layer3.add_module('fc2', nn.Linear(512, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc2_nopool_avg_64(nn.Module):
    def __init__(self):
        super(Conv3fc2_nopool_avg_64, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False))
        layer1.add_module('bn13', nn.BatchNorm2d(256))
        layer1.add_module('relu13', nn.ReLU(True))
        layer1.add_module('pool13', nn.AvgPool2d(14))
        self.layer1 = layer1

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(256, 512))
        layer3.add_module('fc2', nn.Linear(512, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


#输入图像 112 × 112
class Conv3fc2(nn.Module):
    def __init__(self):
        super(Conv3fc2, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False))  # b 32 56 56
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False))  # b 64 28 28
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv21', nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False))  # b 128 14 14
        layer2.add_module('bn21', nn.BatchNorm2d(128))
        layer2.add_module('relu21', nn.ReLU(True))
        layer2.add_module('pool', nn.AvgPool2d(14))  # b 128 1 1
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(128, 256))
        layer3.add_module('fc2', nn.Linear(256, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x

# 输入图像 112 × 112
class Conv3fc1_nopool(nn.Module):
    def __init__(self):
        super(Conv3fc1_nopool, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False))  # b 32 56 56
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False))  # b 64 28 28
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv21', nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False))  # b 128 14 14
        layer2.add_module('bn21', nn.BatchNorm2d(128))
        layer2.add_module('relu21', nn.ReLU(True))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(128*14*14, 2))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


