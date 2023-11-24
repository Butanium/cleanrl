import numpy as np
import torch
from torch import nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def atari_network(orth_init=False):
    init = layer_init if orth_init else lambda m: m
    return nn.Sequential(
        init(nn.Conv2d(6, 32, 8, stride=4)),
        nn.ReLU(),
        init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        init(nn.Linear(64 * 7 * 7, 512)),
        nn.ReLU(),
    )
