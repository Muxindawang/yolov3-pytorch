import torch
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets=None):
    # print images
    # 水平翻转
    images = torch.flip(images, dims=[2])
    # targets[:, 2] = 1 - targets[:, 2]
    # print
    print(images)
    return images, targets

if __name__ == '__main__':
    a = torch.randn((2, 3, 4))
    horisontal_flip(a)
