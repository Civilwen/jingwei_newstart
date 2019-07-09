import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(Residual_block, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1, return_indices=False))
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

    def _make_layer(self, in_channel, out_channel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        layers = []
        layers.append(Residual_block(in_channel, out_channel, stride=stride, shortcut=shortcut))
        for i in range(1, block_num):
            layers.append(Residual_block(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        out_1 = self.pre(x)
        print("pre:", out_1.shape)
        out_2 = self.layer1(out_1)
        print("layer1:", out_2.shape)
        out_3 = self.layer2(out_2)
        print("layer2:", out_3.shape)
        out_4 = self.layer3(out_3)
        print("layer3:", out_4.shape)
        out_5 = self.layer4(out_4)
        print("layer4:", out_5.shape)
        return out_1, out_2, out_3, out_4, out_5


if __name__ == '__main__':
    net = ResNet34()
    x = torch.rand((1, 3, 1024, 1024))
    print(x.shape)
    print(net.forward(x))
