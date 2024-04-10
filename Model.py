#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/4/9 14:29
# @Author : ZhangKuo
import torch.nn as nn
from torchvision import models
from torchinfo import summary


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(
                1
            ),  # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()

        # Inital Convolution  3*256*256 -> 64*256*256
        out_channels = 64
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(
                in_channels
            ),  # padding, keep the image size constant after next conv2d
            nn.Conv2d(in_channels, out_channels, 2 * in_channels + 1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        channels = out_channels

        # Downsampling   64*256*256 -> 128*128*128 -> 256*64*64
        self.down = []
        for _ in range(2):
            out_channels = channels * 2
            self.down += [
                nn.Conv2d(channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            channels = out_channels
        self.down = nn.Sequential(*self.down)

        # Transformation (ResNet)  256*64*64
        self.trans = [ResidualBlock(channels) for _ in range(num_residual_blocks)]
        self.trans = nn.Sequential(*self.trans)

        # Upsampling  256*64*64 -> 128*128*128 -> 64*256*256
        self.up = []
        for _ in range(2):
            out_channels = channels // 2
            self.up += [
                nn.Upsample(scale_factor=2),  # bilinear interpolation
                nn.Conv2d(channels, out_channels, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            channels = out_channels
        self.up = nn.Sequential(*self.up)

        # Out layer  64*256*256 -> 3*256*256
        self.out = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(channels, in_channels, 2 * in_channels + 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        x = self.trans(x)
        x = self.up(x)
        x = self.out(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = models.vit_b_16(weights=None)
        self.norm = nn.LayerNorm(1000)
        self.head = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.model(x)
        x = self.norm(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    des = Discriminator()
    summary(des, input_size=(8, 3, 224, 224))
