#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/4/9 14:21
# @Author : ZhangKuo
from torchvision.transforms import transforms


def get_transforms():
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform
