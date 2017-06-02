import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import OrderedDict
from torch.autograd import Variable

def down_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.BatchNorm2d(out_channel)),
            ("relu1", nn.ReLU()),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.BatchNorm2d(out_channel)),
            ("relu1", nn.ReLU())]))
    return layer

def up_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.BatchNorm2d(out_channel)),
            ("relu1", nn.ReLU()),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.BatchNorm2d(out_channel)),
            ("relu2", nn.ReLU())]))
    return layer

def center_crop(layer, target_size):
    _, _, layer_width, layer_height = layer.size()
    start = (layer_width - target_size) // 2
    crop = layer[:, :, start:(start + target_size), start:(start + target_size)]
    return crop

def concatenate(link, layer):
    crop = center_crop(link, layer.size()[2])
    concat = torch.cat([crop, layer], 1)
    return concat

def hellinger_distance(y_true, y_pred):
    dif = torch.sqrt(y_true) - torch.sqrt(y_pred)
    dif = torch.pow(dif, 2)
    dif = torch.sqrt(torch.sum(dif)) / 1.4142135623730951
    return dif

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        # convolution path
        self.down1 = down_layer(in_channel=1, out_channel=16, kernel_size=3, padding=0)
        self.max1 = nn.MaxPool2d(2)
        self.down2 = down_layer(in_channel=16, out_channel=32, kernel_size=3, padding=0)
        self.max2 = nn.MaxPool2d(2)
        self.down3 = down_layer(in_channel=32, out_channel=64, kernel_size=3, padding=0)
        self.max3 = nn.MaxPool2d(2)
        self.down4 = down_layer(in_channel=64, out_channel=128, kernel_size=3, padding=0)

        # deconvolution path
        self.up3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv3 = up_layer(in_channel=128, out_channel=64, kernel_size=3, padding=0)

        self.up2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up_conv2 = up_layer(in_channel=64, out_channel=32, kernel_size=3, padding=0)

        self.up1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.up_conv1 = up_layer(in_channel=32, out_channel=16, kernel_size=3, padding=0)

        self.last = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        # down
        out = self.down1(x)
        link1 = out
        out = self.max1(out)
        out = self.down2(out)
        link2 = out
        out = self.max2(out)

        out = self.down3(out)
        link3 = out
        out = self.max3(out)

        out = self.down4(out)

        # up
        out = self.up3(out)
        out = concatenate(link3, out)
        out = self.up_conv3(out)

        out = self.up2(out)
        out = concatenate(link2, out)
        out = self.up_conv2(out)

        out = self.up1(out)
        out = concatenate(link1, out)
        out = self.up_conv1(out)
        out = F.sigmoid(out)
        return out

if __name__ == "__main__":
    net = UNET()
    criterion = hellinger_distance
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

