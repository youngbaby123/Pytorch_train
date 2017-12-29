from torch import nn


class simpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class simpleNet_1(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim):
        super(simpleNet_1, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Activation_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Activation_Net_1(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim):
        super(Activation_Net_1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Batch_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Lenet_3(nn.Module):
    def __init__(self):
        super(Lenet_3, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(1, 6, 3, padding=1))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(6, 16, 5))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(400, 120))
        layer3.add_module('fc2', nn.Linear(120, 84))
        layer3.add_module('fc3', nn.Linear(84, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class Lenet_CovAct(nn.Module):
    def __init__(self):
        super(Lenet_CovAct, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(1, 6, 3, padding=1))
        layer1.add_module('conv11', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(6, 16, 5))
        layer2.add_module('conv22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(400, 120))
        layer3.add_module('fc2', nn.Linear(120, 84))
        layer3.add_module('fc3', nn.Linear(84, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class Lenet_AllAct(nn.Module):
    def __init__(self):
        super(Lenet_AllAct, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(1, 6, 3, padding=1))
        layer1.add_module('conv11', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(6, 16, 5))
        layer2.add_module('conv22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(400, 120))
        layer3.add_module('fc11', nn.ReLU(True))
        layer3.add_module('fc2', nn.Linear(120, 84))
        layer3.add_module('fc22', nn.ReLU(True))
        layer3.add_module('fc3', nn.Linear(84, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class Lenet_CovActBN(nn.Module):
    def __init__(self):
        super(Lenet_CovActBN, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(1, 6, 3, padding=1))
        layer1.add_module('bn1', nn.BatchNorm2d(6))
        layer1.add_module('conv11', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(6, 16, 5))
        layer2.add_module('bn2', nn.BatchNorm2d(16))
        layer2.add_module('conv22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(400, 120))
        layer3.add_module('fc11', nn.ReLU(True))
        layer3.add_module('fc2', nn.Linear(120, 84))
        layer3.add_module('fc22', nn.ReLU(True))
        layer3.add_module('fc3', nn.Linear(84, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class Lenet_AllActBN(nn.Module):
    def __init__(self):
        super(Lenet_AllActBN, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(1, 6, 3, padding=1))
        layer1.add_module('bn1', nn.BatchNorm2d(6))
        layer1.add_module('conv11', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(6, 16, 5))
        layer2.add_module('bn2', nn.BatchNorm2d(16))
        layer2.add_module('conv22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(400, 120))
        layer3.add_module('bnfc1', nn.BatchNorm1d(120))
        layer3.add_module('fc11', nn.ReLU(True))
        layer3.add_module('fc2', nn.Linear(120, 84))
        layer3.add_module('bnfc2', nn.BatchNorm1d(84))
        layer3.add_module('fc22', nn.ReLU(True))
        layer3.add_module('fc3', nn.Linear(84, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class Conv2_fc3_ActBN(nn.Module):
    def __init__(self):
        super(Conv2_fc3_ActBN, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(1, 6, 3, padding=1))  # b 6 28 28
        layer1.add_module('bn1', nn.BatchNorm2d(6))
        layer1.add_module('conv11', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))  # b 6 14 14
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(6, 16, 3, padding=1))  # b 16 14 14
        layer2.add_module('bn2', nn.BatchNorm2d(16))
        layer2.add_module('conv22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))  # b 16 7 7
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(16 * 7 * 7, 1024))
        layer3.add_module('bnfc1', nn.BatchNorm1d(1024))
        layer3.add_module('fc11', nn.ReLU(True))
        layer3.add_module('fc2', nn.Linear(1024, 128))
        layer3.add_module('bnfc2', nn.BatchNorm1d(128))
        layer3.add_module('fc22', nn.ReLU(True))
        layer3.add_module('fc3', nn.Linear(128, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class Conv4_fc3_ActBN(nn.Module):
    def __init__(self):
        super(Conv4_fc3_ActBN, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(1, 8, 3, padding=1))  # b 8 28 28
        layer1.add_module('bn11', nn.BatchNorm2d(8))
        layer1.add_module('relu11', nn.ReLU(True))

        layer1.add_module('conv12', nn.Conv2d(8, 16, 3, padding=1))  # b 16 28 28
        layer1.add_module('bn12', nn.BatchNorm2d(16))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))  # b 16 14 14
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv21', nn.Conv2d(16, 32, 3, padding=1))  # b 32 14 14
        layer2.add_module('bn21', nn.BatchNorm2d(32))
        layer2.add_module('relu21', nn.ReLU(True))
        layer2.add_module('conv22', nn.Conv2d(32, 64, 3, padding=1))  # b 64 14 14
        layer2.add_module('bn22', nn.BatchNorm2d(64))
        layer2.add_module('relu22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))  # b 64 7 7
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 7 * 7, 1024))
        layer3.add_module('bnfc1', nn.BatchNorm1d(1024))
        layer3.add_module('relufc1', nn.ReLU(True))
        layer3.add_module('fc2', nn.Linear(1024, 128))
        layer3.add_module('bnfc2', nn.BatchNorm1d(128))
        layer3.add_module('relufc2', nn.ReLU(True))
        layer3.add_module('fc3', nn.Linear(128, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class Conv4_fc3_ActBNdrop(nn.Module):
    def __init__(self):
        super(Conv4_fc3_ActBNdrop, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(1, 8, 3, padding=1))  # b 8 28 28
        layer1.add_module('bn11', nn.BatchNorm2d(8))
        layer1.add_module('relu11', nn.ReLU(True))

        layer1.add_module('conv12', nn.Conv2d(8, 16, 3, padding=1))  # b 16 28 28
        layer1.add_module('bn12', nn.BatchNorm2d(16))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))  # b 16 14 14
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv21', nn.Conv2d(16, 32, 3, padding=1))  # b 32 14 14
        layer2.add_module('bn21', nn.BatchNorm2d(32))
        layer2.add_module('relu21', nn.ReLU(True))
        layer2.add_module('conv22', nn.Conv2d(32, 64, 3, padding=1))  # b 64 14 14
        layer2.add_module('bn22', nn.BatchNorm2d(64))
        layer2.add_module('relu22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))  # b 64 7 7
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 7 * 7, 1024))
        layer3.add_module('bnfc1', nn.BatchNorm1d(1024))
        layer3.add_module('relufc1', nn.ReLU(True))
        layer3.add_module('dropfc1', nn.Dropout2d(0.5))
        layer3.add_module('fc2', nn.Linear(1024, 128))
        layer3.add_module('bnfc2', nn.BatchNorm1d(128))
        layer3.add_module('relufc2', nn.ReLU(True))
        layer3.add_module('dropfc2', nn.Dropout2d(0.5))
        layer3.add_module('fc3', nn.Linear(128, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class Conv4_fc3_ActBN_tiny(nn.Module):
    def __init__(self):
        super(Conv4_fc3_ActBN_tiny, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(1, 8, 3, padding=1))  # b 8 28 28
        layer1.add_module('bn11', nn.BatchNorm2d(8))
        layer1.add_module('relu11', nn.ReLU(True))

        layer1.add_module('conv12', nn.Conv2d(8, 8, 3, padding=1))  # b 8 28 28
        layer1.add_module('bn12', nn.BatchNorm2d(8))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))  # b 16 14 14
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv21', nn.Conv2d(8, 8, 3, padding=1))  # b 8 14 14
        layer2.add_module('bn21', nn.BatchNorm2d(8))
        layer2.add_module('relu21', nn.ReLU(True))
        layer2.add_module('conv22', nn.Conv2d(8, 8, 3, padding=1))  # b 8 14 14
        layer2.add_module('bn22', nn.BatchNorm2d(8))
        layer2.add_module('relu22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))  # b 8 7 7
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(8 * 7 * 7, 128))
        layer3.add_module('bnfc1', nn.BatchNorm1d(128))
        layer3.add_module('relufc1', nn.ReLU(True))
        layer3.add_module('fc2', nn.Linear(128, 64))
        layer3.add_module('bnfc2', nn.BatchNorm1d(64))
        layer3.add_module('relufc2', nn.ReLU(True))
        layer3.add_module('fc3', nn.Linear(64, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class Conv4_fc3_ActBN_fctiny(nn.Module):
    def __init__(self):
        super(Conv4_fc3_ActBN_fctiny, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(1, 8, 3, padding=1))  # b 8 28 28
        layer1.add_module('bn11', nn.BatchNorm2d(8))
        layer1.add_module('relu11', nn.ReLU(True))

        layer1.add_module('conv12', nn.Conv2d(8, 16, 3, padding=1))  # b 16 28 28
        layer1.add_module('bn12', nn.BatchNorm2d(16))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))  # b 16 14 14
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv21', nn.Conv2d(16, 32, 3, padding=1))  # b 32 14 14
        layer2.add_module('bn21', nn.BatchNorm2d(32))
        layer2.add_module('relu21', nn.ReLU(True))
        layer2.add_module('conv22', nn.Conv2d(32, 64, 3, padding=1))  # b 64 14 14
        layer2.add_module('bn22', nn.BatchNorm2d(64))
        layer2.add_module('relu22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))  # b 64 7 7
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 7 * 7, 128))
        layer3.add_module('bnfc1', nn.BatchNorm1d(128))
        layer3.add_module('relufc1', nn.ReLU(True))
        layer3.add_module('fc2', nn.Linear(128, 64))
        layer3.add_module('bnfc2', nn.BatchNorm1d(64))
        layer3.add_module('relufc2', nn.ReLU(True))
        layer3.add_module('fc3', nn.Linear(64, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class Conv4_fc3_ActBN_convtiny(nn.Module):
    def __init__(self):
        super(Conv4_fc3_ActBN_convtiny, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(1, 8, 3, padding=1))  # b 8 28 28
        layer1.add_module('bn11', nn.BatchNorm2d(8))
        layer1.add_module('relu11', nn.ReLU(True))

        layer1.add_module('conv12', nn.Conv2d(8, 8, 3, padding=1))  # b 8 28 28
        layer1.add_module('bn12', nn.BatchNorm2d(8))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))  # b 16 14 14
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv21', nn.Conv2d(8, 8, 3, padding=1))  # b 8 14 14
        layer2.add_module('bn21', nn.BatchNorm2d(8))
        layer2.add_module('relu21', nn.ReLU(True))
        layer2.add_module('conv22', nn.Conv2d(8, 8, 3, padding=1))  # b 8 14 14
        layer2.add_module('bn22', nn.BatchNorm2d(8))
        layer2.add_module('relu22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))  # b 8 7 7
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(8 * 7 * 7, 1024))
        layer3.add_module('bnfc1', nn.BatchNorm1d(1024))
        layer3.add_module('relufc1', nn.ReLU(True))
        layer3.add_module('fc2', nn.Linear(1024, 128))
        layer3.add_module('bnfc2', nn.BatchNorm1d(128))
        layer3.add_module('relufc2', nn.ReLU(True))
        layer3.add_module('fc3', nn.Linear(128, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class Conv4_fc2_ActBN_fctiny(nn.Module):
    def __init__(self):
        super(Conv4_fc2_ActBN_fctiny, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(1, 8, 3, padding=1))  # b 8 28 28
        layer1.add_module('bn11', nn.BatchNorm2d(8))
        layer1.add_module('relu11', nn.ReLU(True))

        layer1.add_module('conv12', nn.Conv2d(8, 16, 3, padding=1))  # b 16 28 28
        layer1.add_module('bn12', nn.BatchNorm2d(16))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))  # b 16 14 14
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv21', nn.Conv2d(16, 32, 3, padding=1))  # b 32 14 14
        layer2.add_module('bn21', nn.BatchNorm2d(32))
        layer2.add_module('relu21', nn.ReLU(True))
        layer2.add_module('conv22', nn.Conv2d(32, 64, 3, padding=1))  # b 64 14 14
        layer2.add_module('bn22', nn.BatchNorm2d(64))
        layer2.add_module('relu22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))  # b 64 7 7
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 7 * 7, 128))
        layer3.add_module('bnfc1', nn.BatchNorm1d(128))
        layer3.add_module('relufc1', nn.ReLU(True))
        layer3.add_module('fc3', nn.Linear(128, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class Conv4_fc1_ActBN_fctiny(nn.Module):
    def __init__(self):
        super(Conv4_fc1_ActBN_fctiny, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(1, 8, 3, padding=1))  # b 8 28 28
        layer1.add_module('bn11', nn.BatchNorm2d(8))
        layer1.add_module('relu11', nn.ReLU(True))

        layer1.add_module('conv12', nn.Conv2d(8, 16, 3, padding=1))  # b 16 28 28
        layer1.add_module('bn12', nn.BatchNorm2d(16))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))  # b 16 14 14
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv21', nn.Conv2d(16, 32, 3, padding=1))  # b 32 14 14
        layer2.add_module('bn21', nn.BatchNorm2d(32))
        layer2.add_module('relu21', nn.ReLU(True))
        layer2.add_module('conv22', nn.Conv2d(32, 64, 3, padding=1))  # b 64 14 14
        layer2.add_module('bn22', nn.BatchNorm2d(64))
        layer2.add_module('relu22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))  # b 64 7 7
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64 * 7 * 7, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


class Conv4_conv1_ActBN(nn.Module):
    def __init__(self):
        super(Conv4_conv1_ActBN, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(1, 8, 3, padding=1))  # b 8 28 28
        layer1.add_module('bn11', nn.BatchNorm2d(8))
        layer1.add_module('relu11', nn.ReLU(True))

        layer1.add_module('conv12', nn.Conv2d(8, 16, 3, padding=1))  # b 16 28 28
        layer1.add_module('bn12', nn.BatchNorm2d(16))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))  # b 16 14 14
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv21', nn.Conv2d(16, 32, 3, padding=1))  # b 32 14 14
        layer2.add_module('bn21', nn.BatchNorm2d(32))
        layer2.add_module('relu21', nn.ReLU(True))
        layer2.add_module('conv22', nn.Conv2d(32, 64, 3, padding=1))  # b 64 14 14
        layer2.add_module('bn22', nn.BatchNorm2d(64))
        layer2.add_module('relu22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))  # b 64 7 7
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv31', nn.Conv2d(64, 10, 7))  # b 10 1 1
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        return x


class Conv4_conv2_ActBN(nn.Module):
    def __init__(self):
        super(Conv4_conv2_ActBN, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(1, 8, 3, padding=1))  # b 8 28 28
        layer1.add_module('bn11', nn.BatchNorm2d(8))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(8, 16, 3, padding=1))  # b 16 28 28
        layer1.add_module('bn12', nn.BatchNorm2d(16))
        layer1.add_module('relu12', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2))  # b 16 14 14
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv21', nn.Conv2d(16, 32, 3, padding=1))  # b 32 14 14
        layer2.add_module('bn21', nn.BatchNorm2d(32))
        layer2.add_module('relu21', nn.ReLU(True))
        layer2.add_module('conv22', nn.Conv2d(32, 64, 3, padding=1))  # b 64 14 14
        layer2.add_module('bn22', nn.BatchNorm2d(64))
        layer2.add_module('relu22', nn.ReLU(True))
        layer2.add_module('poo2', nn.MaxPool2d(2, 2))  # b 64 7 7
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv31', nn.Conv2d(64, 32, 7))  # b 32 1 1
        layer3.add_module('bn31', nn.BatchNorm2d(32))
        layer3.add_module('relu31', nn.ReLU(True))
        layer3.add_module('conv32', nn.Conv2d(32, 10, 1))  # b 10 1 1
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        return x


# class MobileNet(nn.Module):
#     def __init__(self):
#         super(MobileNet, self).__init__()
#
#         def conv_bn(inp, oup, stride):
#             return nn.Sequential(
#                 nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU(inplace=True)
#             )
#
#         def conv_dw(inp, oup, stride):
#             return nn.Sequential(
#                 nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#                 nn.BatchNorm2d(inp),
#                 nn.ReLU(inplace=True),
#
#                 nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU(inplace=True),
#             )
#
#         self.model = nn.Sequential(
#             conv_bn(1, 32, 1),      #28
#             conv_dw(32, 64, 1),     #28
#             conv_dw(64, 128, 2),    #14
#             conv_dw(128, 128, 1),   #14
#             conv_dw(128, 256, 1),   #7
#             conv_dw(256, 256, 1),
#             conv_dw(256, 512, 1),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 1024, 2),
#             conv_dw(1024, 1024, 1),
#             nn.AvgPool2d(7),
#         )
#         self.fc = nn.Linear(1024, 10)
#
#     def forward(self, x):
#         x = self.model(x)
#         x = x.view(-1, 1024)
#         x = self.fc(x)
#         return x

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

class Conv3fc2(nn.Module):
    def __init__(self):
        super(Conv3fc2, self).__init__()

        layer1 = nn.Sequential()
        layer1.add_module('conv11', nn.Conv2d(3, 32, 3, stride=2, padding=1))  # b 32 56 56
        layer1.add_module('bn11', nn.BatchNorm2d(32))
        layer1.add_module('relu11', nn.ReLU(True))
        layer1.add_module('conv12', nn.Conv2d(32, 64, 3, stride=2, padding=1))  # b 64 28 28
        layer1.add_module('bn12', nn.BatchNorm2d(64))
        layer1.add_module('relu12', nn.ReLU(True))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv21', nn.Conv2d(64, 128, 3, stride=2, padding=1))  # b 128 14 14
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

class MobileNet_conv3(nn.Module):
    def __init__(self):
        super(MobileNet_conv3, self).__init__()

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


class MobileNet_conv3(nn.Module):
    def __init__(self):
        super(MobileNet_conv3, self).__init__()

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
