import torch.nn as nn
import torch
from torchvision import transforms

from utils.math import *
from models.mlp_policy import Policy

# borrows some ideas from https://github.com/astooke/rlpyt/blob/master/rlpyt/models/conv2d.py


# means and stds from imagenet
imgnet_means = [0.485, 0.456, 0.406]
imgnet_stds = [0.229, 0.224, 0.225]

def img_transform(img_means, img_stds):
    return transforms.Compose((
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_means, std=img_stds)
    ))


def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
    sh, sw = stride if isinstance(stride, tuple) else (stride,) * 2
    ph, pw = padding if isinstance(padding, tuple) else (padding,) * 2
    d = dilation
    h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
    return h, w


class CNNBasic(nn.Module):
    def __init__(self, state_dim, action_dim, channels, kernel_sizes, strides, paddings=None,
                 activation='relu', use_maxpool=False, num_aux=0, resnet_first_layer=False):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]

        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)

        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
            self.activation_mod = nn.Tanh
        elif activation == 'relu':
            self.activation = torch.relu
            self.activation_mod = nn.ReLU
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
            self.activation = nn.Sigmoid

        h, w, in_channels = state_dim

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
            sequence.extend([conv_layer, self.activation_mod()])
            if maxp_stride > 1:
                sequence.append(nn.MaxPool2d(maxp_stride))  # No padding.

        self.conv = nn.Sequential(*sequence)

        # set up simple FC head, can be overwritten
        self.conv_out_size_for_fc = self.conv_out_size(h, w)
        self.head = nn.Linear(self.conv_out_size_for_fc + num_aux, action_dim)
        self.num_aux = num_aux

        # initialize first cnn layer using resnet101 weights,
        # first layer must be nn.Conv2d(3, 64, kernel_size=7, stride=4)
        if resnet_first_layer:
            self.conv[0].apply(self.prior_init)

    def forward(self, img, aux_states=None):
        conv_out = self.conv(img).view(img.shape[0], -1)
        if aux_states is None:
            assert self.num_aux == 0
            return self.head(conv_out)
        else:
            assert self.num_aux == aux_states.shape[1]
            return self.head(torch.cat([conv_out, aux_states], 1))

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

    def prior_init(self, m):
        if type(m) == nn.Conv2d:
            # initialize the first rgb layer using googlenet or similar
            import os, pickle
            full_path = os.path.dirname(os.path.realpath(__file__))
            with open(full_path + '/../assets/cafferesnet_layer1_weights.pkl', 'rb') as file:
                layer1_weights = pickle.load(file)
            m.weight.data = layer1_weights