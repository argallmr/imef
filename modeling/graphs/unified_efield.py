import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
import download_data as dd
import data_manipulation as dm
import argparse
import storage_objects
import datetime as dt
import xarray as xr

#TODO: Generalize inputs so that it isn't just Kp
# Make error for not enough data points
# Make comments and remove unneeded code
# Save the model, and then make a new program for testing new data with said model
# Remove testing from here, and just put that part of the code in the new testing program to be made


class TestNeuralNetwork(nn.Module):
    def __init__(self):
        super(TestNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # First number of first thing has to be 63 (# of datapoints)
            # Last number of last thing is  3 (Efield in x, y, and z directions)
            # The other two numbers are arbitrary.
            nn.Linear(63, 20),
            nn.Sigmoid(),
            nn.Linear(20, 12),
            nn.Sigmoid(),
            nn.Linear(12, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def get_inputs(data_filename):
    imef_data = xr.open_dataset(data_filename)
    index_data = imef_data['Kp']
    imef_data = imef_data.where(np.isnan(imef_data['E_GSE'][:, 0]) == False, drop=True)

    for counter in range(60, len(imef_data['time'].values)):
        wanted_index_data = index_data.values[counter - 60:counter].tolist()
        the_rest_of_the_data = np.array([imef_data['L'].values[counter], np.cos(imef_data['MLT'].values[counter]),
                                         np.sin(imef_data['MLT'].values[counter])]).tolist()
        new_data_line = wanted_index_data + the_rest_of_the_data
        if counter == 60:
            design_matrix_array = [new_data_line]
        else:
            design_matrix_array.append(new_data_line)

    design_matrix_array = torch.tensor(design_matrix_array)
    efield_data = torch.from_numpy(imef_data['E_GSE'].values[60:, :])

    return design_matrix_array, efield_data


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def main():
    parser = argparse.ArgumentParser(
        description='PUT DESCRIPTION HERE'
    )
    #
    parser.add_argument('input_filename', type=str, help='File name of the data created by sample_data.py. Include file extension')

    parser.add_argument('input_filename_2', type=str, help='probably dont keep this for real')
    #
    # parser.add_argument('filename', type=str, help='Output file name. Do not include file extension')
    #
    # parser.add_argument('-p', '--polar', help='Convert the electric field values to polar (default is cartesian)',
    #                     action='store_true')
    #
    args = parser.parse_args()

    # For now I will train on everything. Though whether this should actually be done will have to be revisited
    # Should I train on other mms probes or just mms1?

    train_filename = args.input_filename
    test_filename = args.input_filename_2

    # design_matrix = torch.from_numpy(design_matrix_array)

    train_inputs, train_targets = get_inputs(train_filename)
    test_inputs, test_targets = get_inputs(test_filename)

    # Im going to jump. Everything I had done to this point was not needed. Pytorch has it's own thing for creating a tensor dataset
    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    # need to do training and test somehow. Will probably have to do something about that in IMEFDataset.
    # Or rework it so that IMEF dataset takes the data I give it instead of reading the file directly.
    # Wait I only need training data for this program. I could just use the 6 years of data as training and use the data from 9/1/21 onwards as test data.
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    model = TestNeuralNetwork().to(device)

    # All this stuff is what will have to be messed with in order to get best possible approximation. Along with the layers in the network itself
    # how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
    learning_rate = 1e-3
    # the number of data samples propagated through the network before the parameters are updated
    batch_size = 1
    # the number times to iterate over the dataset
    epochs = 10
    # The loss function
    loss_fn = nn.MSELoss()
    # Something else to be messed with idk
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")


if __name__ == '__main__':
    main()