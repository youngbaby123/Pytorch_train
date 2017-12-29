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