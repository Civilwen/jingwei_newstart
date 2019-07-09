import torch
import torch.nn as nn
import resnet34
import torch.nn.functional as F
import torchvision.models as models


class Decode_block(nn.Module):
    def __init__(self, in_channel, out_channel, dconv_kernel, dconv_stride):
        super(Decode_block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channel, out_channel // 2, kernel_size=dconv_kernel, stride=dconv_stride),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.dconv(x)
        return out


class Unet(nn.Module):
    def __init__(self, out_channel):
        super(Unet, self).__init__()
        # self.in_channel = in_channel
        self.out_channel = out_channel
        # self.downsample = resnet34.ResNet34()
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dconv_5 = Decode_block(512, 512, 2, 2)
        self.dconv_4 = Decode_block(512, 256, 2, 2)
        self.dconv_3 = Decode_block(256, 128, 2, 2)
        self.dconv_2 = Decode_block(128, 64, 2, 2)
        self.dconv_1 = Decode_block(128, 64, 8, 4)

        self.final_out = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channel, 1, 1)
        )

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        out_1 = self.firstmaxpool(x)

        out_2 = self.encoder1(out_1)
        out_3 = self.encoder2(out_2)
        out_4 = self.encoder3(out_3)
        out_5 = self.encoder4(out_4)
        # out_1, out_2, out_3, out_4, out_5 = self.downsample(x)

        d_out_4 = self.dconv_5(out_5)
        d_out_3 = self.dconv_4(torch.cat([out_4, d_out_4], 1))
        d_out_2 = self.dconv_3(torch.cat([out_3, d_out_3], 1))
        d_out_1 = self.dconv_1(torch.cat([out_2, d_out_2], 1))

        out = self.final_out(d_out_1)

        return out


def Unet_ResNet34(out_channel, requires_gard=True):
    model = Unet(out_channel)
    model.require_encoder_grad(requires_grad=requires_gard)

    return model


if __name__ == '__main__':
    x = torch.rand((1, 3, 1024, 1024))
    net = Unet(4)
    net.require_encoder_grad(requires_grad=True)
    a = net.forward(x)
    print(a.shape)
