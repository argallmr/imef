import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import xarray as xr
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import Neural_Networks as NN

# For debugging purposes
np.set_printoptions(threshold=np.inf)


def get_inputs(imef_data, remove_nan=True, get_target_data=True, use_values=['All'], usetorch=True):
    # This could be made way more efficient if I were to make the function not download all the data even if it isn't used. But for sake of understandability (which this has little of anyways)
    # I leave it this way. Also when I get Sym-H and start needing to do combos I'm gonna have to make use_values a list and split, etc

    # if use_values != 'All' and use_values != 'Kp' and use_values != 'Dst':
    #     raise KeyError('The use_values argument must either be \'All\', or any combination of: Kp, Dst, Sym-h')
    if use_values[0] == 'All':
        Kp_data = imef_data['Kp']
        Dst_data = imef_data['DST']
        # Symh_data = imef_data['SYMH']
    else:
        if 'Kp' in use_values:
            Kp_data = imef_data['Kp']
        if 'Dst' in use_values:
            Dst_data = imef_data['DST']
        if 'Symh' in use_values:
            Symh_data = imef_data['SYMH']
    if remove_nan == True:
        imef_data = imef_data.where(np.isnan(imef_data['E_GSE'][:, 0]) == False, drop=True)

    # Note that the first bits of data cannot be used, because the first 60 times dont have Kp values and whatnot. Will become negligible when done on a large amount of data
    for counter in range(60, len(imef_data['time'].values)):
        Kp_index_data = Kp_data.values[counter - 60:counter].tolist()
        Dst_index_data = Dst_data.values[counter - 60:counter].tolist()
        # Symh_index_data = Symh_data.values[counter - 60:counter].tolist()

        the_rest_of_the_data = np.array([imef_data['L'].values[counter], np.cos(np.pi/12*imef_data['MLT'].values[counter]), np.sin(np.pi/12*imef_data['MLT'].values[counter])]).tolist()

        if use_values[0]=='All':
            # new_data_line = Kp_index_data + Dst_index_data + Symh_index_data + the_rest_of_the_data
            new_data_line = Kp_index_data + Dst_index_data + the_rest_of_the_data
        else:
            new_data_line=[]
            if 'Kp' in use_values:
                new_data_line += Kp_index_data
            if 'Dst' in use_values:
                new_data_line += Dst_index_data
            # if 'Symh' in use_values:
            #     new_data_line += Symh_index_data
            new_data_line += the_rest_of_the_data


        if counter == 60:
            design_matrix_array = [new_data_line]
        else:
            # FOR TESTING PURPOSES. SHOULD NOT KEEP HERE FOR ANY REAL RUNS
            # if counter > 150 and counter <= 882:
            #     pass
            # else:
            #     design_matrix_array.append(new_data_line)
            design_matrix_array.append(new_data_line)

    if usetorch==True:
        design_matrix_array = torch.tensor(design_matrix_array)
    else:
        design_matrix_array = np.array(design_matrix_array)

    if get_target_data == True:
        efield_data = imef_data['E_GSE'].values[60:, :]
        # AGAIN REMOVE THIS TOO
        # efield_data = np.concatenate((efield_data[0:150], efield_data[882:]))
        if usetorch==True:
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


def train_NN(args):
    train_filename_list = args.input_filename.split(',')
    values_to_use = args.input_list
    layers = args.layers
    model_name = args.model_filename + '.pth'
    random = args.random
    kfold = args.kfold
    adam=args.adam

    if ',' in values_to_use:
        values_to_use = values_to_use.split(',')
    else:
        values_to_use = [values_to_use]

    for train_filename in train_filename_list:
        total_data = xr.open_dataset(train_filename+'.nc')
        one_file_inputs, one_file_targets = get_inputs(total_data, use_values=values_to_use)
        if train_filename == train_filename_list[0]:
            total_inputs = one_file_inputs
            total_targets = one_file_targets
        else:
            total_inputs = torch.concat((total_inputs, one_file_inputs))
            total_targets = torch.concat((total_targets, one_file_targets))

    # total_inputs = total_inputs[150:882]
    # total_targets=total_targets[150:882]

    batch_size = 32
    # batch_size =1

    kf = KFold(n_splits=5, shuffle=True, random_state=142)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    NN_dict = {1: NN.NeuralNetwork_1,
               2: NN.NeuralNetwork_2,
               3: NN.NeuralNetwork_3}

    # HERE IS WHERE TO DETERMINE LAYER ARCHITECTURE
    # change this to be 183 when symh is involved
    if values_to_use[0] == 'All':
        NN_layout = np.array([123])
    else:
        NN_layout = np.array([60*len(values_to_use)+3])

    if random == True:
        random_values = np.array(layers.split(','))
        NN_layout = np.append(NN_layout, np.random.randint(random_values[0], random_values[1], random_values[2]))
    else:
        NN_layout = np.append(NN_layout, np.array(layers.split(",")))
    NN_layout = np.append(NN_layout, np.array([3])).astype(int)
    number_of_layers = len(NN_layout)-2

    try:
        NeuralNetwork = NN_dict[number_of_layers]
    except:
        raise KeyError("The amount of layers inputted is not available")

    model = NeuralNetwork(NN_layout).to(device)

    # To output to a txt file, which will be used for keeping track of the NN's I run and their errors
    counter=0
    values_string = ''
    for value in values_to_use:
        values_string += value
    string = str('Inputs: ' + values_string + ' || Layers: ')
    for parameter in model.parameters():
        if counter%2==0 and counter < 2*number_of_layers-2:
            string = string + str(len(parameter)) + '-'
        elif counter%2 == 0 and counter == 2*number_of_layers-2:
            string = string + str(len(parameter))
        counter+=1

    # All this stuff is what will have to be messed with in order to get best possible approximation. Along with the layers in the network itself
    # how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
    learning_rate = .99    # the number times to iterate over the dataset
    epochs = 100
    loss_fn = nn.MSELoss()
    # Something else to be messed with idk
    if adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # The actual training
    final_test_error = np.zeros((3))
    if kfold:
        for train_index, test_index in kf.split(total_inputs):
            # I think that this does reset the NN, so it can retrain from scratch. Should test a little more tho just in case, but I can't find a way to do so
            # If I feel the need to change this, make a list containing 5 models before the for loop (one for each fold), and then pick a model from there, and use each for each pass through the loop
            model = NeuralNetwork(NN_layout).to(device)

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
    else:
        model = NeuralNetwork(NN_layout).to(device)

        train_inputs = total_inputs
        train_targets = total_targets

        train_dataset = TensorDataset(train_inputs, train_targets)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer, device)

    final_test_error = final_test_error/(10*kf.get_n_splits(total_inputs))
    print("Done!")

    # Output the properties and the test results of the NN
    put_error_here = open('test_errors.txt', 'a')
    output = string + str(' || ExMSE: '+ str(final_test_error[0])+ ' || EyMSE: '+str(final_test_error[1])+ ' || EzMSE: '
                 +str(final_test_error[2])+ ' || Total E MSE: '+ str(np.sum(final_test_error))+'\n')
    put_error_here.write(output)

    print(output)

    torch.save(model.state_dict(), model_name)


def linear_regression(train_filename_list, values_to_use):
    for train_filename in train_filename_list:
        total_data = xr.open_dataset(train_filename+'.nc')
        one_file_inputs, one_file_targets = get_inputs(total_data, use_values=values_to_use, usetorch=False)
        if train_filename == train_filename_list[0]:
            total_inputs = one_file_inputs
            total_targets = one_file_targets
        else:
            total_inputs = np.concat((total_inputs, one_file_inputs))
            total_targets = np.concat((total_targets, one_file_targets))

    kf = KFold(n_splits=5, shuffle=True, random_state=142)
    test_error = np.zeros([3])
    for train_index, test_index in kf.split(total_inputs):
        train_inputs, test_inputs = total_inputs[train_index], total_inputs[test_index]
        train_targets, test_targets = total_targets[train_index], total_targets[test_index]

        # is this resetting the training? I'm mostly confident it is, but can't really check definitively
        LR = LinearRegression()
        LR.fit(train_inputs, train_targets)

        length = len(test_inputs[0])

        for counter in range(len(test_inputs)):
            pred = LR.predict(test_inputs[counter].reshape(-1, length))[0]
            error = np.sqrt(pred**2 + test_targets[counter]**2)
            test_error += error

    test_error /= len(total_targets)

    print("Done!")

    values_string = ''
    for value in values_to_use:
        values_string += value

    string = str('Inputs: ' + values_string + ' || Layers: Linear Regression')
    # Output the properties and the test results of the model
    put_error_here = open('test_errors.txt', 'a')
    output = string + str(' || ExMSE: ' + str(test_error[0]) + ' || EyMSE: ' + str(test_error[1]) + ' || EzMSE: '
            + str(test_error[2]) + ' || Total E MSE: ' + str(np.sum(test_error)) + '\n')
    put_error_here.write(output)

    print(output)

def main():
    parser = argparse.ArgumentParser(
        description='PUT DESCRIPTION HERE'
    )
    #
    parser.add_argument('input_filename', type=str, help='File name(s) of the data created by sample_data.py. If more than 1 file, use the format filename1,filename2,filename3 ... '
                                                         'Do not include file extension')

    parser.add_argument('input_list', type=str, help='Name(s) of the indices you want to be used in the NN. Options are: Kp, Dst, and All')

    parser.add_argument('layers', type=str, help='The number of nodes you want in each layer of your NN. Eg, a 3 layer NN would look something like 30,20,15. If you want to use linear regression, type LR instead')

    parser.add_argument('model_filename', type=str, help='Desired output name of the file containing the trained NN. Do not include file extension. Note that nothing will be saved for linear regression'
                                                         'Also note that if you want to pass the model into other files in this package, you should name the file in the following format:'
                                                         'Layer1-Layer2-LayerX$input_list$input_filename. Eg: 50-20$Kp,Dst$090115_033022_mms1_sample_data')

    parser.add_argument('-r', '--random', help='Randomize the amount of nodes in each hidden layer of the created NN. If you choose to do this, you must input x,y,z as your layers argument, '
                                               'where x is the lower bound on the number of nodes, y is the upper bound, and z is the number of layers. '
                                               'Eg: 15,50,3 will make a NN with 3 layers, and all 3 layers have a random number of nodes ranging from 15 to 50. No effect on linear regression', action='store_true')

    parser.add_argument('-k', '--kfold' , help='If the user wants to run K-fold validation, type -tr. Note that the last fold will be saved as the model, and not a model trained on all the data', action='store_true')

    parser.add_argument('-a', '--adam', help='Use ADAM as the optimizer instead of the default Stochastic Gradient Descent',
                        action='store_true')

    args = parser.parse_args()

    train_filename_list = args.input_filename.split(',')
    values_to_use = [args.input_list]
    layers=args.layers

    if layers == 'LR':
        linear_regression(train_filename_list, values_to_use)
    else:
        train_NN(args)

if __name__ == '__main__':
    main()