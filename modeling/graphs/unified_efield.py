import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import xarray as xr
from sklearn.model_selection import KFold
import Neural_Networks as NN

# For debugging purposes
np.set_printoptions(threshold=np.inf)

# TODO: Implement cross-validation, so that test errors of NN's are more accurate
#  Replace number of layers and random with an argument that takes a list of numbers and turns that into a NN

def get_inputs(imef_data, remove_nan=True, get_target_data=True, remove_dup=False, use_values='All'):
    # This could be made way more efficient if I were to make the function not download all the data even if it isn't used. But for sake of understandability (which this has little of anyways)
    # I leave it this way. Also when I get Sym-H and start needing to do combos I'm gonna have to make use_values a list and split, etc

    if use_values != 'All' and use_values != 'Kp' and use_values != 'Dst':
        raise KeyError('The use_values argument must be one of: All, Kp, Dst')

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

        if use_values=='Kp':
            new_data_line = Kp_index_data + the_rest_of_the_data
        elif use_values=='Dst':
            new_data_line = Dst_index_data + the_rest_of_the_data
        else:
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
    test_loss= np.zeros((3))

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            test_loss += np.array([loss_fn(pred[:,0], y[:,0]).item(), loss_fn(pred[:,1], y[:,1]).item(), loss_fn(pred[:,2], y[:,2]).item()])

    test_loss /= num_batches
    print("Test Error:")
    print('   Avg MSE in Ex: ', test_loss[0])
    print('   Avg MSE in Ey: ', test_loss[1])
    print('   Avg MSE in Ez: ', test_loss[2])
    print('   Avg MSE in all components: ', np.sum(test_loss), '\n')

    return test_loss


def main():
    parser = argparse.ArgumentParser(
        description='PUT DESCRIPTION HERE'
    )
    #
    parser.add_argument('input_filename', type=str, help='File name(s) of the data created by sample_data.py. If more than 1 file, use the format filename1,filename2,filename3 ... '
                                                         'Do not include file extension')

    parser.add_argument('input_list', type=str, help='Name(s) of the indices you want to be used in the NN. Options are: Kp, Dst, and All')

    parser.add_argument('layers', type=str, help='The number of nodes you want in each layer of your NN. Eg, a 3 layer NN would look something like 30,20,15. If you want linear regression, type LR instead')

    parser.add_argument('model_filename', type=str, help='Desired output name of the file containing the trained NN. Do not include file extension')

    parser.add_argument('-r', '--random', help='Randomize the amount of nodes in each hidden layer of the created NN', action='store_true')

    args = parser.parse_args()

    train_filename_list = args.input_filename.split(',')
    values_to_use = args.input_list
    # number_of_layers = args.number_of_layers
    layers=args.layers
    model_name = args.model_filename+'.pth'
    random = args.random

    for train_filename in train_filename_list:
        total_data = xr.open_dataset(train_filename+'.nc')
        one_file_inputs, one_file_targets = get_inputs(total_data, use_values=values_to_use)
        if train_filename == train_filename_list[0]:
            total_inputs = one_file_inputs
            total_targets = one_file_targets
        else:
            total_inputs = torch.concat((total_inputs, one_file_inputs))
            total_targets = torch.concat((total_targets, one_file_targets))

    # HERE IS WHERE TO DETERMINE LAYER ARCHITECTURE


    # Take the file and take a portion of it as test data, use rest as train data
    # train_inputs, test_inputs, train_targets, test_targets = train_test_split(total_inputs, total_targets, test_size=.2)

    batch_size = 32

    kf = KFold(n_splits=5, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    NN_dict = {0: NN.Linear_Regression,
               1: NN.NeuralNetwork_1,
               2: NN.NeuralNetwork_2,
               3: NN.NeuralNetwork_3}

    try:
        NeuralNetwork = NN_dict[number_of_layers]
    except:
        raise KeyError("The amount of layers inputted is not available")

    if values_to_use == 'All':
        model = NeuralNetwork(123, random).to(device)
    else:
        model = NeuralNetwork(63, random).to(device)

    counter=0
    if number_of_layers != 0:
        string = str('Inputs: ' + values_to_use + ' || Layers: ')
        for parameter in model.parameters():
            if counter%2==0 and counter < 2*number_of_layers-2:
                string = string + str(len(parameter)) + '-'
            elif counter%2 == 0 and counter == 2*number_of_layers-2:
                string = string + str(len(parameter))
            counter+=1
    else:
        string = str('Inputs: ' + values_to_use + ' || Linear Regression ')

    # All this stuff is what will have to be messed with in order to get best possible approximation. Along with the layers in the network itself
    # how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
    # 1e-2 seems good for NNs, but for linear regression it seems to not work. 1e-5 worked, tho. IDK how big of a deal this is
    learning_rate = 1e-2
    # the number times to iterate over the dataset
    epochs = 1000
    loss_fn = nn.MSELoss()
    # Something else to be messed with idk
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # The actual training
    final_test_error = np.zeros((3))
    for train_index, test_index in kf.split(total_inputs):
        train_inputs, test_inputs = total_inputs[train_index], total_inputs[test_index]
        train_targets, test_targets = total_targets[train_index], total_targets[test_index]

        train_dataset = TensorDataset(train_inputs, train_targets)
        test_dataset = TensorDataset(test_inputs, test_targets)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer, device)
            test_error = test_loop(test_dataloader, model, loss_fn, device)
            if epochs-t <=10: # This may be something to remove. May be best to only use last test error after the dataset gets cleaned of outliers
                final_test_error+=test_error

    final_test_error = final_test_error/(10*kf.get_n_splits(total_inputs))
    print("Done!")

    # Output the test results of the NN and the
    put_error_here = open('test_errors.txt', 'a')
    output = string + str(' || ExMSE: '+ str(final_test_error[0])+ ' || EyMSE: '+str(final_test_error[1])+ ' || EzMSE: '
                 +str(final_test_error[2])+ ' || Total E MSE: '+ str(np.sum(final_test_error))+'\n')
    put_error_here.write(output)

    print(output)

    torch.save(model.state_dict(), model_name)

if __name__ == '__main__':
    main()