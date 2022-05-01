from torch import nn
import numpy as np


# Doesn't work. IDK why
# but it does work when # inputs = # outputs, which is strange
# note that random isn't needed in LN, since there are no hidden layers. It is just left there since otherwise it would be more annoying to deal with in unified_efield
class Linear_Regression(nn.Module):
    def __init__(self, number_of_inputs, random=False):
        super(Linear_Regression, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # First number of first layer has to be # of inputs
            # Last number of last layer is 3 (Efield in x, y, and z directions)
            # The other numbers are arbitrary. Mess around with them and see what happens
            nn.Linear(number_of_inputs, 3),
        )

    def forward(self, x):
        out = self.linear(x)
        return out


class NeuralNetwork_1(nn.Module):
    def __init__(self, number_of_inputs, random=False):
        super(NeuralNetwork_1, self).__init__()
        self.flatten = nn.Flatten()
        if random == True:
            first_layer_count = np.random.randint(5, 100)
        else:
            first_layer_count = 20
        self.linear_relu_stack = nn.Sequential(
            # First number of first layer has to be # of inputs
            # Last number of last layer is 3 (Efield in x, y, and z directions)
            # The other numbers are arbitrary. Mess around with them and see what happens
            nn.Linear(number_of_inputs, first_layer_count),
            nn.Sigmoid(),
            nn.Linear(first_layer_count, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork_2(nn.Module):
    def __init__(self, number_of_inputs, random):
        super(NeuralNetwork_2, self).__init__()
        self.flatten = nn.Flatten()
        if random == True:
            first_layer_count = np.random.randint(5, 100)
            second_layer_count = np.random.randint(5, 100)
        else:
            first_layer_count = 20
            second_layer_count = 10
        self.linear_relu_stack = nn.Sequential(
            # First number of first layer has to be # of inputs
            # Last number of last layer is 3 (Efield in x, y, and z directions)
            # The other numbers are arbitrary. Mess around with them and see what happens
            nn.Linear(number_of_inputs, first_layer_count),
            nn.Sigmoid(),
            nn.Linear(first_layer_count, second_layer_count),
            nn.Sigmoid(),
            nn.Linear(second_layer_count, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork_3(nn.Module):
    def __init__(self, number_of_inputs, random):
        super(NeuralNetwork_3, self).__init__()
        self.flatten = nn.Flatten()
        if random == True:
            first_layer_count = np.random.randint(5, 100)
            second_layer_count = np.random.randint(5, 100)
            third_layer_count = np.random.randint(5, 100)
        else:
            first_layer_count = 20
            second_layer_count = 10
            third_layer_count = 15
        self.linear_relu_stack = nn.Sequential(
            # First number of first layer has to be # of inputs
            # Last number of last layer is 3 (Efield in x, y, and z directions)
            # The other numbers are arbitrary. Mess around with them and see what happens
            nn.Linear(number_of_inputs, first_layer_count),
            nn.Sigmoid(),
            nn.Linear(first_layer_count, second_layer_count),
            nn.Sigmoid(),
            nn.Linear(second_layer_count, third_layer_count),
            nn.Sigmoid(),
            nn.Linear(third_layer_count, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits