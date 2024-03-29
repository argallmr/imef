from torch import nn
from dataclasses import dataclass
import numpy as np


@dataclass
class NN_parameters(object):
    NN_layout: type(np.array)
    batch_size: int
    learning_rate: float
    epochs: int
    kfold: int
    random_undersampling: int
    # these two are not supposed to be strings, but I don't know how to not set an expected type of object
    loss_fn: str
    optimizer: str


class NeuralNetwork_1(nn.Module):
    def __init__(self, layers):
        super(NeuralNetwork_1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # First number of first layer has to be # of inputs
            # Last number of last layer is 3 (Efield in x, y, and z directions)
            # The other numbers are arbitrary. Mess around with them and see what happens
            nn.Linear(layers[0], layers[1]),
            nn.Sigmoid(),
            nn.Linear(layers[1], layers[2]),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork_2(nn.Module):
    def __init__(self, layers):
        super(NeuralNetwork_2, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # First number of first layer has to be # of inputs
            # Last number of last layer is 3 (Efield in x, y, and z directions)
            # The other numbers are arbitrary. Mess around with them and see what happens
            nn.Linear(layers[0], layers[1]),
            nn.Sigmoid(),
            nn.Linear(layers[1], layers[2]),
            nn.Sigmoid(),
            nn.Linear(layers[2], layers[3]),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork_3(nn.Module):
    def __init__(self, layers):
        super(NeuralNetwork_3, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # First number of first layer has to be # of inputs
            # Last number of last layer is 3 (Efield in x, y, and z directions)
            # The other numbers are arbitrary. Mess around with them and see what happens
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
            nn.ReLU(),
            nn.Linear(layers[2], layers[3]),
            nn.ReLU(),
            nn.Linear(layers[3], layers[4]),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits