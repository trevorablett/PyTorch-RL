import torch.nn as nn
import torch
from utils.math import *
from models.mlp_policy import Policy
from models.cnn_common import CNNBasic
from models.mlp_discriminator import Discriminator


class CNNDiscriminator(CNNBasic):
    def __init__(self, state_dim, action_dim, channels, kernel_sizes, strides, paddings=None,
                 head_hidden_size=(128, 128), num_aux=0, activation='relu', use_maxpool=False,
                 resnet_first_layer=False):
        super().__init__(state_dim, 1, channels, kernel_sizes, strides, paddings,
                         activation, use_maxpool, num_aux, resnet_first_layer)

        self.head = Discriminator(self.conv_out_size_for_fc + action_dim + num_aux,
                                  head_hidden_size, activation)

    def forward(self, img, action, aux_states=None):
        conv_out = self.conv(img).view(img.shape[0], -1)
        if aux_states is None:
            assert self.num_aux == 0
            return self.head(torch.cat([conv_out, action], 1))
        else:
            assert self.num_aux == aux_states.shape[1]
            return self.head(torch.cat([conv_out, action, aux_states], 1))