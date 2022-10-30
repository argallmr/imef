import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import xarray as xr
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import imef.data.data_manipulation as dm
import pickle
import imef.efield.model_creation.NN_functions as NN_func

# For debugging purposes
np.set_printoptions(threshold=np.inf)


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
    layers = args.layers.split(',')
    model_name = args.model_filename + '.pth'
    random = args.random
    kfold = args.kfold
    adam=args.adam
    undersample = args.random_undersampling
    quiet_storm_ratio = args.quiet_storm_ratio

    # Create a list containing all the geomagnetic indices the user wants
    if values_to_use=='All':
        values_to_use = ['Kp', 'Dst', 'Symh']
    elif ',' in values_to_use:
        values_to_use = values_to_use.split(',')
    else:
        values_to_use = [values_to_use]

    if undersample == True:
        undersample_ratio = quiet_storm_ratio
    else:
        undersample_ratio = None

    # For every file given by the user, open it, gather the necessary inputs that will train the neural network, and create one object containing all of that data
    for train_filename in train_filename_list:
        one_file = xr.open_dataset(train_filename+'.nc')
        one_file_inputs, one_file_targets = dm.get_NN_inputs(one_file, use_values=values_to_use, undersample=undersample_ratio)
        if train_filename == train_filename_list[0]:
            total_inputs = one_file_inputs
            total_targets = one_file_targets
        else:
            total_inputs = torch.concat((total_inputs, one_file_inputs))
            total_targets = torch.concat((total_targets, one_file_targets))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    number_of_layers = len(layers)
    NN_layout = np.array([NN_func.get_predictor_counts(number_of_layers)])

    if random == True:
        random_values = np.array(layers)
        NN_layout = np.append(NN_layout, np.random.randint(random_values[0], random_values[1], random_values[2]))
    else:
        NN_layout = np.append(NN_layout, np.array(layers))
    NN_layout = np.append(NN_layout, np.array([3])).astype(int)

    model=NN_func.get_NN(NN_layout, device=device)

    # These values are all things to be messed with to get the optimal neural network.
    # the number of points
    batch_size = args.batch_size
    # The amount each batch of points will affect the training of the NN
    learning_rate = args.learning_rate
    # the number times to iterate over the dataset
    epochs = args.epochs
    # The function that the model wants to optimize
    # Finding some other useful papers on different loss functions could be crucial, as it was suggested to try piecewise functions to help predict the nonlinear nature of the electric field
    loss_fn = nn.MSELoss()
    # an option for optimizers. Generally adam is always used, as it tends to be the best option, but I leave stochastic gradient descent as the default
    if adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # The actual training
    # Note that if k-fold is not done the final test error will be 0
    final_test_error = np.zeros((3))
    if kfold:
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(total_inputs):
            # I think that this does reset the NN, so it can retrain from scratch. Should test a little more tho just in case, but I can't find a way to do so
            # If I feel the need to change this, make a list containing 5 models before the for loop (one for each fold), and then pick a model from there, and use each for each pass through the loop
            model = NN_func.get_NN(NN_layout, device=device)

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

        final_test_error = final_test_error / (10 * kf.get_n_splits(total_inputs))

        print("Done!")
        output = NN_func.output_error(values_to_use, model.parameters(), number_of_layers, final_test_error, file_to_output_to = 'test_errors.txt')

        print(output)
    else:
        model = NN_func.get_NN(NN_layout, device=device)

        train_inputs = total_inputs
        train_targets = total_targets

        train_dataset = TensorDataset(train_inputs, train_targets)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer, device)

        print("Done!")

    # create a file name that contains the important components of the model
    layer_string = str(NN_layout[1])+'-'
    for counter in range(2, len(NN_layout)-2):
        layer_string = layer_string + str(NN_layout[counter])+'-'
    layer_string+=str(NN_layout[len(NN_layout)-2])
    if undersample == True:
        model_name = 'undersampled$'+model_name
    values_string = ''
    for value in values_to_use:
        values_string += value
    model_name = layer_string+'$'+values_string+'$'+model_name

    # save the model
    torch.save(model.state_dict(), model_name)


def linear_regression(train_filename_list, values_to_use, kfold, model_filename):
    if values_to_use=='All':
        values_to_use = ['Kp', 'Dst', 'Symh']
    elif ',' in values_to_use:
        values_to_use = values_to_use.split(',')
    else:
        values_to_use = [values_to_use]

    # For every file given by the user, open it, gather the necessary inputs that will train the neural network, and create one object containing all of that data
    for train_filename in train_filename_list:
        total_data = xr.open_dataset(train_filename+'.nc')
        one_file_inputs, one_file_targets = dm.get_NN_inputs(total_data, use_values=values_to_use, usetorch=False)
        if train_filename == train_filename_list[0]:
            total_inputs = one_file_inputs
            total_targets = one_file_targets
        else:
            total_inputs = np.concat((total_inputs, one_file_inputs))
            total_targets = np.concat((total_targets, one_file_targets))

    if kfold:
        kf = KFold(n_splits=5, shuffle=True, random_state=142)
        test_error = np.zeros([3])
        total=0
        for train_index, test_index in kf.split(total_inputs):
            train_inputs, test_inputs = total_inputs[train_index], total_inputs[test_index]
            train_targets, test_targets = total_targets[train_index], total_targets[test_index]

            # is this resetting the training? I'm mostly confident it is, but can't really check definitively
            LR = LinearRegression()
            LR.fit(train_inputs, train_targets)

            length = len(test_inputs[0])

            for counter in range(len(test_inputs)):
                total+=1
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
    else:
        print('fitting')
        LR = LinearRegression()
        LR.fit(total_inputs, total_targets)
        print("Done!")

    values_name_string = ''
    for strvalue in values_to_use:
        if strvalue == values_to_use[-1]:
            values_name_string=values_name_string+strvalue+'$'
        else:
            values_name_string = values_name_string + strvalue + '-'
    filename = 'LR$'+values_name_string+model_filename

    pickle.dump(LR, open(filename, 'wb'))

def main():
    parser = argparse.ArgumentParser(
        description='This program trains a neural network to predict the convection electric field at a certain time and location, using location data and geomagnetic indices.'
    )

    parser.add_argument('input_filename', type=str, help='File name(s) of the data created by sample_data.py. If more than 1 file, use the format filename1,filename2,filename3 ... '
                                                         'Do not include file extension')

    parser.add_argument('input_list', type=str, help='Name(s) of the indices you want to be used in the NN. Options are: Kp, Dst, Symh, and All')

    parser.add_argument('layers', type=str, help='The number of nodes you want in each layer of your NN. Eg, a 3 layer NN would look something like 30,20,15. If you want to use linear regression, type LR instead')

    parser.add_argument('model_filename', type=str, help='Desired output name of the file containing the trained model. Do not include file extension.'
                                                         'Also note that if you want to pass the model into other files in this package, you should name the file in the following format:'
                                                         'Layer1-Layer2-LayerX$input_list$input_filename. Eg: 50-20$Kp,Dst$090115_033022_mms1_sample_data')

    parser.add_argument('-r', '--random', help='Randomize the amount of nodes in each hidden layer of the created NN. If you choose to do this, you must input x,y,z as your layers argument, '
                                               'where x is the lower bound on the number of nodes, y is the upper bound, and z is the number of layers. '
                                               'Eg: 15,50,3 will make a NN with 3 layers, and all 3 layers have a random number of nodes ranging from 15 to 50. No effect on linear regression', action='store_true')

    parser.add_argument('-k', '--kfold', help='If the user wants to run K-fold validation, type -k. Note that the model trained on the last fold will be saved, and not a model trained on all the data', action='store_true')

    parser.add_argument('-a', '--adam', help='Use ADAM as the optimizer instead of the default Stochastic Gradient Descent',
                        action='store_true')

    parser.add_argument('-bs', '--batch_size',
                        default=64,
                        type=int,
                        help='The batch size defines the number of samples that will be propagated through the network.',
                        )

    parser.add_argument('-lr', '--learning_rate',
                        default=.01,
                        type=float,
                        help='# The amount each batch of points will affect the training of the NN',
                        )

    parser.add_argument('-e', '--epochs',
                        default=1000,
                        type=int,
                        help='the number times to iterate over the dataset',
                        )

    parser.add_argument('-ru', '--random_undersampling', help='Randomly undersample the data file given, such that the quiet time data and storm time data are made more balanced',
                        action='store_true',
                        )

    parser.add_argument('-qsr', '--quiet_storm_ratio',
                        default=1,
                        type=float,
                        help='Ratio of quiet versus storm time data. Default is 1. Does nothing if -ru is not used',
                        )

    args = parser.parse_args()

    train_filename_list = args.input_filename.split(',')
    values_to_use = args.input_list
    layers=args.layers
    model_filename = args.model_filename
    kfold=args.kfold

    if layers == 'LR':
        linear_regression(train_filename_list, values_to_use, kfold, model_filename)
    else:
        train_NN(args)

if __name__ == '__main__':
    main()