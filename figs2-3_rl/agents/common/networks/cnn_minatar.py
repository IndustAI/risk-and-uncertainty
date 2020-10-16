"""
Code derived from the MinAtar github repo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


class CNNMinAtar(nn.Module):
    def __init__(self, observation_space, n_outputs, weight_scale=np.sqrt(2)):

        super().__init__()
        self.weight_scale = weight_scale

        if len(observation_space.shape) != 3:
            raise NotImplementedError

        in_channels = observation_space.shape[2]

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=n_outputs)

        self.conv.apply(lambda x: init_weights(x, self.weight_scale))
        self.fc_hidden.apply(lambda x: init_weights(x, self.weight_scale))
        self.output.apply(lambda x: init_weights(x, self.weight_scale))

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(0)

        x = x.permute(0, 3, 1, 2)

        # Rectified output from the first conv layer
        x = F.relu(self.conv(x))

        # Rectified output from the final hidden layer
        x = F.relu(self.fc_hidden(x.contiguous().view(x.size(0), -1)))

        # Returns the output from the fully-connected linear layer
        return self.output(x)


class CNNMinAtar_k_head(nn.Module):
    def __init__(self, observation_space, n_outputs, n_heads=10, weight_scale=np.sqrt(2)):

        super().__init__()
        self.weight_scale = weight_scale
        self.n_heads = n_heads
        self.n_outputs = n_outputs

        if len(observation_space.shape) != 3:
            raise NotImplementedError

        in_channels = observation_space.shape[2]

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

        self.output = nn.Sequential(
            nn.Conv1d(num_linear_units * n_heads, 128 * n_heads, kernel_size=1, groups=n_heads),
            nn.ReLU(),
            nn.Conv1d(128 * n_heads, n_outputs * n_heads, kernel_size=1, groups=n_heads),
        )

        self.conv.apply(lambda x: init_weights(x, self.weight_scale))
        self.output.apply(lambda x: init_weights(x, self.weight_scale))

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(0)

        x = x.permute(0, 3, 1, 2)

        # Rectified output from the first conv layer
        x = F.relu(self.conv(x))

        x = x.contiguous().view(x.size(0), -1) # Flatten the conv output new shape : B x 128
        x = x.repeat(1, self.n_heads).unsqueeze(2)

        return self.output(x).reshape(-1, self.n_heads, self.n_outputs)
