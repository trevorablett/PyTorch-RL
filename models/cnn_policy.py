import torch.nn as nn
import torch
from utils.math import *
from models.mlp_policy import Policy, StochPolicyBasic
from models.cnn_common import CNNBasic


class CNNPolicy(CNNBasic, StochPolicyBasic):  # inheritance from StochPolicyBasic gives select_action, etc.
    def __init__(self, state_dim, action_dim, channels, kernel_sizes, strides, paddings=None,
                 head_hidden_size=(128, 128), num_aux=0,
                 activation='relu', use_maxpool=False, log_std=0):
        super().__init__(state_dim, action_dim, channels, kernel_sizes, strides, paddings,
                         activation, use_maxpool, num_aux)

        self.head = Policy(self.conv_out_size_for_fc + num_aux, action_dim, head_hidden_size, activation, log_std)