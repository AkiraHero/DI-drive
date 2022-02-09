import torch
from torch import nn
from torchvision.models.resnet import resnet18


def gen_up(inchannels, outchannels, scale):
    return nn.Sequential(
        nn.Conv2d(inchannels, outchannels, 3, padding=1, bias=False),
        nn.BatchNorm2d(outchannels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=scale)
    )


class CNN(nn.Module):
    def __init__(self, cin, cout,
                 with_skip, dropout_p):
        super(CNN, self).__init__()

        self.cin = cin
        self.cout = cout
        self.trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.trunk.conv1 = nn.Conv2d(cin, 64, kernel_size=7,
                                     stride=2, padding=3,
                                     bias=False)
        self.downscale = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2,
                      padding=1, bias=False),
            )
        self.upscale0 = gen_up(512, 512, scale=2)
        self.upscale1 = gen_up(512, 256, scale=4)

        if with_skip:
            self.upscale2 = gen_up(256+128, 256, scale=2)
            self.upscale3 = gen_up(256+64, 128, scale=4)
        else:
            self.upscale2 = gen_up(256, 256, scale=2)
            self.upscale3 = gen_up(256, 128, scale=4)
        self.with_skip = with_skip
        self.final = nn.Conv2d(128, cout, 1)
        self.dropout = nn.Dropout(p=dropout_p, inplace=False)

    def forward(self, x):
        x = self.trunk_forward(x)
        return x

    def trunk_forward(self, x):
        x = self.trunk.conv1(x)
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x)

        x1 = self.trunk.layer1(x)
        x1 = self.dropout(x1)

        x2 = self.trunk.layer2(x1)
        x2 = self.dropout(x2)

        x = self.trunk.layer3(x2)
        x = self.dropout(x)

        x = self.trunk.layer4(x)
        x = self.downscale(x)

        x = self.upscale0(x)
        x = self.upscale1(x)

        if self.with_skip:
            x = self.upscale2(torch.cat((x, x2), 1))
            x = self.upscale3(torch.cat((x, x1), 1))
        else:
            x = self.upscale2(x)
            x = self.upscale3(x)

        x = self.final(x)

        return x


if __name__ == '__main__':
    model = CNN(5, 16, True, 0.0)
    pp = torch.rand([1, 5, 256, 256])
    tt = model(pp)
    pass