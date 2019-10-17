import torch.nn as nn
import torch
from utils.math import *
from models.mlp_policy import Policy

# borrows some ideas from https://github.com/astooke/rlpyt/blob/master/rlpyt/models/conv2d.py


class CNNPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, channels, kernel_sizes, strides, paddings=None,
                 head_hidden_size=(128, 128), activation='relu', use_maxpool=False, log_std=0):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]

        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)

        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        # set up CNN
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [nn.Conv2d(in_channels=ic, out_channels=oc,
                                       kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
                       zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.extend([conv_layer, nonlinearity()])
            if maxp_stride > 1:
                sequence.append(nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = nn.Sequential(*sequence)

        # set up FC head
        c, h, w = state_dim

    def conv_out_size(self, h, w, c=None):
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                                           child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return h * w * c