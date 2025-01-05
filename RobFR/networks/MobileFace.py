import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from RobFR.networks.FaceModel import FaceModel


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # Pointwise
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # Depthwise
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # Pointwise linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, padding, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.prelu = nn.PReLU(oup) if not linear else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x if self.linear else self.prelu(x)


Mobilefacenet_bottleneck_setting = [
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]


class MobileFacenet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFacenet, self).__init__()
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        self.blocks = self._make_layer(Bottleneck, bottleneck_setting)
        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, (7, 7), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 512, 1, 1, 0, linear=True)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, settings):
        layers = []
        for t, c, n, s in settings:
            for i in range(n):
                layers.append(block(self.inplanes, c, s if i == 0 else 1, t))
                self.inplanes = c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = (x - 127.5) / 128.0  # Normalize input
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        return x.view(x.size(0), -1)


class MobileFace(FaceModel):
    def __init__(self, **kwargs):
        net = MobileFacenet()
        url = 'http://ml.cs.tsinghua.edu.cn/~dingcheng/ckpts/face_models/model7/Backbone_mobileface_Epoch_36_Batch_409392_Time_2019-04-07-16-40_checkpoint.pth'
        channel = 'bgr'
        super(MobileFace, self).__init__(net=net, url=url, channel=channel, **kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileFace(device=device)
    dummy_input = torch.randn(1, 3, 112, 112).to(device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
