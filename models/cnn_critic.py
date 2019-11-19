import torch.nn as nn
import torch
from utils.math import *
from models.mlp_policy import Policy
from models.cnn_common import CNNBasic
from models.mlp_critic import Value


class CNNValue(CNNBasic):
    def __init__(self, state_dim, channels, kernel_sizes, strides, paddings=None,
                 head_hidden_size=(128, 128), num_aux=0, activation='relu', use_maxpool=False,
                 resnet_first_layer=False):
        super().__init__(state_dim, 1, channels, kernel_sizes, strides, paddings,
                         activation, use_maxpool, num_aux, resnet_first_layer)

        self.head = Value(self.conv_out_size_for_fc + num_aux, head_hidden_size, activation)