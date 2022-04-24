import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import argparse
import xarray as xr
from sklearn.model_selection import train_test_split

# For debugging purposes
np.set_printoptions(threshold=np.inf)

# Should I train on everything or just MMS1? How would I store multiple probe information in 1 thingy? I would probably have to run sample_data multiple times, and run get_inputs on each file
# Then combine the data together after it has been converted to an array (np or otherwise)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # First number of first layer has to be # of inputs
            # Last number of last layer is 3 (Efield in x, y, and z directions)
            # The other numbers are arbitrary. Mess around with them and see what happens
            nn.Linear(123, 10),
            nn.Sigmoid(),
            nn.Linear(10, 20),
            nn.Sigmoid(),
            nn.Linear(20, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def get_inputs(imef_data, remove_nan=True, get_target_data=True, remove_dup=False):
    # to test: removing duplicate values from Kp and Dst. if the are 3 or 6 values (respectively), then remove the first. new NN would have 3+2+5=10 inputs. Maybe prevent overfitting
    Kp_data = imef_data['Kp']
    Dst_data = imef_data['DST']
    if remove_nan == True:
        imef_data = imef_data.where(np.isnan(imef_data['E_GSE'][:, 0]) == False, drop=True)

    # Note that the first bits of data cannot be used, because the first 60 times dont have Kp values and whatnot. Will become negligible when done on a large amount of data
    for counter in range(60, len(imef_data['time'].values)):
        if remove_dup == True:
            # this takes away the 5 minute bins and only puts the values of the 2 Kp values and 5 Dst values (3 and 6 gets harder, so I'm gonna stick with this)
            Kp_all_data = Kp_data.values[counter - 60:counter].tolist()
            Dst_all_data = Dst_data.values[counter - 60:counter].tolist()
            Kp_index_data = [Kp_all_data[-1], Kp_all_data[-37]]
            Dst_index_data = [Dst_all_data[-1], Dst_all_data[-13], Dst_all_data[-25], Dst_all_data[-37], Dst_all_data[-49]]
        else:
            Kp_index_data = Kp_data.values[counter - 60:counter].tolist()
            Dst_index_data = Dst_data.values[counter - 60:counter].tolist()

        the_rest_of_the_data = np.array([imef_data['L'].values[counter], np.cos(np.pi/12*imef_data['MLT'].values[counter]), np.sin(np.pi/12*imef_data['MLT'].values[counter])]).tolist()
        new_data_line = Kp_index_data + Dst_index_data + the_rest_of_the_data
        if counter == 60:
            design_matrix_array = [new_data_line]
        else:
            design_matrix_array.append(new_data_line)

    design_matrix_array = torch.tensor(design_matrix_array)
    if get_target_data == True:
        efield_data = imef_data['E_GSE'].values[60:, :]
        efield_data = torch.from_numpy(efield_data)

        return design_matrix_array, efield_data
    else:
        return design_matrix_array


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X.float())
        loss = loss_fn(pred, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss= 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def main():
    parser = argparse.ArgumentParser(
        description='PUT DESCRIPTION HERE'
    )
    #
    parser.add_argument('input_filename', type=str, help='File name(s) of the data created by sample_data.py. If more than 1 file, use the format filename1,filename2,filename3 ... '
                                                         'Do not include file extension')

    parser.add_argument('model_filename', type=str, help='Desired output name of the file containing the trained NN. Do not include file extension')

    args = parser.parse_args()

    train_filename_list = args.input_filename.split(',')

    model_name = args.model_filename+'.pth'

    for train_filename in train_filename_list:
        total_data = xr.open_dataset(train_filename+'.nc')
        one_file_inputs, one_file_targets = get_inputs(total_data)
        if train_filename == train_filename_list[0]:
            total_inputs = one_file_inputs
            total_targets = one_file_targets
        else:
            total_inputs = torch.concat((total_inputs, one_file_inputs))
            total_targets = torch.concat((total_targets, one_file_targets))


    # Take the file and take a portion of it as test data, use rest as train data
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(total_inputs, total_targets, test_size=.15)

    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    batch_size = 32

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    model = NeuralNetwork().to(device)

    # All this stuff is what will have to be messed with in order to get best possible approximation. Along with the layers in the network itself
    # how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
    learning_rate = 1e-2
    # the number times to iterate over the dataset
    epochs = 1000
    loss_fn = nn.MSELoss()
    # Something else to be messed with idk
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # The actual training
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
    print("Done!")

    torch.save(model.state_dict(), model_name)

if __name__ == '__main__':
    main()