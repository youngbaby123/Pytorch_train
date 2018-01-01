# -*- coding: utf-8 -*-
from torch import nn


# -*- coding: utf-8 -*-
from torch import nn


def conv_bn_3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_bn_5(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 5, stride, 2, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_bn_7(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 7, stride, 3, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw_3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

def conv_dw_5(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 5, stride, 2, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

def conv_dw_7(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 7, stride, 3, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

#输入图像 224 × 224
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

#输入图像大小 112*112
class MobileNet_dw8(nn.Module):
    def __init__(self):
        super(MobileNet_dw8, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),      # b, 32, 56, 56
            conv_dw(32, 64, 1),     # b, 64, 56, 56
            conv_dw(64, 128, 2),    # b, 128, 28, 28
            conv_dw(128, 128, 1),   # b, 128, 28, 28
            conv_dw(128, 256, 2),   # b, 256, 14, 14
            conv_dw(256, 256, 1),   # b, 256, 14, 14
            conv_dw(256, 512, 2),   # b, 512, 7, 7
            conv_dw(512, 512, 1),   # b, 512, 7, 7
            conv_dw(512, 512, 1),   # b, 512, 7, 7
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

#输入图像大小 56 × 56
class MobileNet_dw5(nn.Module):
    def __init__(self):
        super(MobileNet_dw5, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),      # b, 32, 28, 28
            conv_dw(32, 64, 1),     # b, 64, 28, 28
            conv_dw(64, 128, 2),    # b, 128, 14, 14
            conv_dw(128, 128, 1),   # b, 128, 14, 14
            conv_dw(128, 256, 2),   # b, 256, 7, 7
            conv_dw(256, 256, 1),   # b, 256, 7, 7
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

#输入图像大小 28 × 28
class MobileNet_dw3(nn.Module):
    def __init__(self):
        super(MobileNet_dw3, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),      # b, 32, 14, 14
            conv_dw(32, 64, 1),     # b, 64, 14, 14
            conv_dw(64, 128, 2),    # b, 128, 7, 7
            conv_dw(128, 128, 1),   # b, 128, 7, 7
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x

#输入图像大小 112 × 112
class DwNet112_dw3_avg_16(nn.Module):
    def __init__(self):
        super(DwNet112_dw3_avg_16, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 16, 2),        # b, 16, 56, 56
            conv_dw_3(16, 32, 2),       # b, 32, 28, 28
            conv_dw_3(32, 64, 2),       # b, 64, 14, 14
            conv_dw_3(64, 128, 2),      # b, 128, 7, 7
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw3_avg_32(nn.Module):
    def __init__(self):
        super(DwNet112_dw3_avg_32, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 32, 2),  # b, 16, 56, 56
            conv_dw_3(32, 64, 2),  # b, 32, 28, 28
            conv_dw_3(64, 128, 2),  # b, 64, 14, 14
            conv_dw_3(128, 256, 2),  # b, 128, 7, 7
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw3_avg_64(nn.Module):
    def __init__(self):
        super(DwNet112_dw3_avg_64, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 64, 2),  # b, 16, 56, 56
            conv_dw_3(64, 128, 2),  # b, 32, 28, 28
            conv_dw_3(128, 256, 2),  # b, 64, 14, 14
            conv_dw_3(256, 512, 2),  # b, 128, 7, 7
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw3_max_16(nn.Module):
    def __init__(self):
        super(DwNet112_dw3_max_16, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 16, 2),  # b, 16, 56, 56
            conv_dw_3(16, 32, 2),  # b, 32, 28, 28
            conv_dw_3(32, 64, 2),  # b, 64, 14, 14
            conv_dw_3(64, 128, 2),  # b, 128, 7, 7
            nn.MaxPool2d(7),
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw3_max_32(nn.Module):
    def __init__(self):
        super(DwNet112_dw3_max_32, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 32, 2),  # b, 16, 56, 56
            conv_dw_3(32, 64, 2),  # b, 32, 28, 28
            conv_dw_3(64, 128, 2),  # b, 64, 14, 14
            conv_dw_3(128, 256, 2),  # b, 128, 7, 7
            nn.MaxPool2d(7),
        )
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw3_max_64(nn.Module):
    def __init__(self):
        super(DwNet112_dw3_max_64, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 64, 2),  # b, 16, 56, 56
            conv_dw_3(64, 128, 2),  # b, 32, 28, 28
            conv_dw_3(128, 256, 2),  # b, 64, 14, 14
            conv_dw_3(256, 512, 2),  # b, 128, 7, 7
            nn.MaxPool2d(7),
        )
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw3_nopool_16(nn.Module):
    def __init__(self):
        super(DwNet112_dw3_nopool_16, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 16, 2),  # b, 16, 56, 56
            conv_dw_3(16, 32, 2),  # b, 32, 28, 28
            conv_dw_3(32, 64, 2),  # b, 64, 14, 14
            conv_dw_3(64, 128, 2),  # b, 128, 7, 7
        )
        self.fc = nn.Linear(128*7*7, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128*7*7)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw3_nopool_32(nn.Module):
    def __init__(self):
        super(DwNet112_dw3_nopool_32, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 32, 2),  # b, 16, 56, 56
            conv_dw_3(32, 64, 2),  # b, 32, 28, 28
            conv_dw_3(64, 128, 2),  # b, 64, 14, 14
            conv_dw_3(128, 256, 2),  # b, 128, 7, 7
        )
        self.fc = nn.Linear(256*7*7, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256*7*7)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw3_nopool_64(nn.Module):
    def __init__(self):
        super(DwNet112_dw3_nopool_64, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 64, 2),  # b, 16, 56, 56
            conv_dw_3(64, 128, 2),  # b, 32, 28, 28
            conv_dw_3(128, 256, 2),  # b, 64, 14, 14
            conv_dw_3(256, 512, 2),  # b, 128, 7, 7
        )
        self.fc = nn.Linear(512*7*7, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512*7*7)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw7_nopool_16(nn.Module):
    def __init__(self):
        super(DwNet112_dw7_nopool_16, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 16, 2),
            conv_dw_3(16, 16, 1),
            conv_dw_3(16, 32, 2),
            conv_dw_3(32, 32, 1),
            conv_dw_3(32, 64, 2),
            conv_dw_3(64, 64, 1),
            conv_dw_3(64, 128, 2),
            conv_dw_3(128, 128, 1),
        )
        self.fc = nn.Linear(128*7*7, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128*7*7)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw7_nopool_32(nn.Module):
    def __init__(self):
        super(DwNet112_dw7_nopool_32, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 32, 2),
            conv_dw_3(32, 32, 1),
            conv_dw_3(32, 64, 2),
            conv_dw_3(64, 64, 1),
            conv_dw_3(64, 128, 2),
            conv_dw_3(128, 128, 1),
            conv_dw_3(128, 256, 2),
            conv_dw_3(256, 256, 1),
        )
        self.fc = nn.Linear(256*7*7, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256*7*7)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw7_nopool_64(nn.Module):
    def __init__(self):
        super(DwNet112_dw7_nopool_64, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 64, 2),
            conv_dw_3(64, 64, 1),
            conv_dw_3(64, 128, 2),
            conv_dw_3(128, 128, 1),
            conv_dw_3(128, 256, 2),
            conv_dw_3(256, 256, 1),
            conv_dw_3(256, 512, 2),
            conv_dw_3(512, 512, 1),
        )
        self.fc = nn.Linear(512*7*7, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512*7*7)
        x = self.fc(x)
        return x


# 输入图像大小 112 × 112
class DwNet112_dw7_avg_16(nn.Module):
    def __init__(self):
        super(DwNet112_dw7_avg_16, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 16, 2),
            conv_dw_3(16, 16, 1),
            conv_dw_3(16, 32, 2),
            conv_dw_3(32, 32, 1),
            conv_dw_3(32, 64, 2),
            conv_dw_3(64, 64, 1),
            conv_dw_3(64, 128, 2),
            conv_dw_3(128, 128, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw7_avg_32(nn.Module):
    def __init__(self):
        super(DwNet112_dw7_avg_32, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 32, 2),
            conv_dw_3(32, 32, 1),
            conv_dw_3(32, 64, 2),
            conv_dw_3(64, 64, 1),
            conv_dw_3(64, 128, 2),
            conv_dw_3(128, 128, 1),
            conv_dw_3(128, 256, 2),
            conv_dw_3(256, 256, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw7_avg_64(nn.Module):
    def __init__(self):
        super(DwNet112_dw7_avg_64, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 64, 2),
            conv_dw_3(64, 64, 1),
            conv_dw_3(64, 128, 2),
            conv_dw_3(128, 128, 1),
            conv_dw_3(128, 256, 2),
            conv_dw_3(256, 256, 1),
            conv_dw_3(256, 512, 2),
            conv_dw_3(512, 512, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw7_max_16(nn.Module):
    def __init__(self):
        super(DwNet112_dw7_max_16, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 16, 2),
            conv_dw_3(16, 16, 1),
            conv_dw_3(16, 32, 2),
            conv_dw_3(32, 32, 1),
            conv_dw_3(32, 64, 2),
            conv_dw_3(64, 64, 1),
            conv_dw_3(64, 128, 2),
            conv_dw_3(128, 128, 1),
            nn.MaxPool2d(7),
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw7_max_32(nn.Module):
    def __init__(self):
        super(DwNet112_dw7_max_32, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 32, 2),
            conv_dw_3(32, 32, 1),
            conv_dw_3(32, 64, 2),
            conv_dw_3(64, 64, 1),
            conv_dw_3(64, 128, 2),
            conv_dw_3(128, 128, 1),
            conv_dw_3(128, 256, 2),
            conv_dw_3(256, 256, 1),
            nn.MaxPool2d(7),
        )
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

# 输入图像大小 112 × 112
class DwNet112_dw7_max_64(nn.Module):
    def __init__(self):
        super(DwNet112_dw7_max_64, self).__init__()

        self.model = nn.Sequential(
            conv_bn_3(3, 64, 2),
            conv_dw_3(64, 64, 1),
            conv_dw_3(64, 128, 2),
            conv_dw_3(128, 128, 1),
            conv_dw_3(128, 256, 2),
            conv_dw_3(256, 256, 1),
            conv_dw_3(256, 512, 2),
            conv_dw_3(512, 512, 1),
            nn.MaxPool2d(7),
        )
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

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
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False))
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
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False))
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
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False))
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
        layer1.add_module('conv11', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn11', nn.BatchNorm2d(64))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False))
        layer1.add_module('bn12', nn.BatchNorm2d(128))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('conv13', nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False))
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


