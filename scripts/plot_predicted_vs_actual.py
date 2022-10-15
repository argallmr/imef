import numpy as np
import xarray as xr
import argparse
import torch
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import imef.efield.model_creation.Neural_Networks as NN
from imef.data.data_manipulation import get_NN_inputs, get_storm_intervals
from sklearn.linear_model import LinearRegression
import pickle

def pred_w_NN(data, model, values_to_use):
    test_inputs = get_NN_inputs(data, use_values=values_to_use, remove_nan=False, get_target_data=False)
    model.eval()
    x = test_inputs
    with torch.no_grad():
        pred = model(x)
    return pred


def train_and_pred_w_linear_regression(dip, data, values_to_use):
    # this takes way too long on large datasets. Hence why the user has to train and provide an LR model instead of training here
    total_inputs, total_targets = get_NN_inputs(data, use_values=values_to_use, usetorch=False, remove_nan=False)
    train_inputs_full = np.concatenate((total_inputs[0:dip[0]], total_inputs[dip[1]:len(total_inputs)-1]))
    train_targets_full = np.concatenate((total_targets[0:dip[0]], total_targets[dip[1]:len(total_targets)-1]))
    train_targets = train_targets_full[np.isnan(train_targets_full[:,0])==False]
    train_inputs = train_inputs_full[np.isnan(train_targets_full[:, 0]) == False]
    test_inputs = total_inputs[dip[0]:dip[1]]
    LR = LinearRegression()
    LR.fit(train_inputs, train_targets)

    length = len(test_inputs[0])

    LR_predicted = np.array([[]])
    for counter in range(len(test_inputs)):
        pred = LR.predict(test_inputs[counter].reshape(-1, length))[0]
        if counter == 0:
            LR_predicted = [pred]
        else:
            LR_predicted = np.concatenate((LR_predicted, [pred]), axis=0)

    return LR_predicted

def pred_w_linear_regression(data, model, values_to_use):
    test_inputs = get_NN_inputs(data, use_values=['Kp', 'Dst'], usetorch=False, remove_nan=False, get_target_data=False)

    length = len(test_inputs[0])

    LR_predicted = np.array([[]])
    for counter in range(len(test_inputs)):
        pred = model.predict(test_inputs[counter].reshape(-1, length))[0]
        if counter == 0:
            LR_predicted = [pred]
        else:
            LR_predicted = np.concatenate((LR_predicted, [pred]), axis=0)

    return LR_predicted


def plot_dips(dips, data, NN_model, values_to_use, mode='Dst', LR_model=None):
    for start, end in dips:
        # slice the data so we only have the ones in the dip
        begin_time = data['time'].values[start]
        end_time = data['time'].values[end]
        intermediate_step = data.where(data['time'] >= begin_time, drop=True)
        sliced_data = intermediate_step.where(intermediate_step['time'] < end_time, drop=True)

        # create a line plot of the dst values during the given dip
        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
        fig.tight_layout()
        register_matplotlib_converters()
        ax1=axes[0,0]
        ax1.set_ylim([-150, 50])
        ax1.set_xlabel('Date')
        ax1.set_ylabel(mode)
        ax1.plot(data['time'].values[start:end], data[mode].values[start:end])
        my_xticks = ax1.get_xticks()
        ax1.set_xticks([my_xticks[0], my_xticks[-1]])

        # Plot the MMS electric field
        ax2=axes[0,1]
        E_data = np.sqrt(data['E_con'].values[start:end,0]**2 + data['E_con'].values[start:end,1]**2 + data['E_con'].values[start:end,0]**2)
        ax2.plot(data['time'].values[start:end], E_data, label='MMS Data')
        ax2.set_ylim([0, 10])
        ax2.set_xlabel('Date')
        ax2.set_ylabel('||E||')

        # predict and plot the electric field using the given neural network
        NN_predicted = pred_w_NN(sliced_data, NN_model, values_to_use)
        # The +3600 is for the first couple points removed from get_NN_values. It may be a good idea to introduce something in get_NN_values to avoid this
        ax2.plot(data['time'].values[start+3600:end],np.sqrt(NN_predicted[:, 0] ** 2 + NN_predicted[:, 1] ** 2 + NN_predicted[:, 2] ** 2),label='Neural Network')

        if LR_model != None:
            # predict and plot the electric field using the given linear regression model
            LR_predicted = pred_w_linear_regression(sliced_data, LR_model, values_to_use)
            ax2.plot(data['time'].values[start+3600:end],np.sqrt(LR_predicted[:, 0] ** 2 + LR_predicted[:, 1] ** 2 + LR_predicted[:, 2] ** 2),label='Linear Regression')

        my_xticks = ax2.get_xticks()
        ax2.set_xticks([my_xticks[0], my_xticks[-1]])
        ax2.legend()

        plt.show()

    return dips

def main():
    parser = argparse.ArgumentParser(
        description='PUT DESCRIPTION HERE'
    )

    # I don't think inputting multiple files here makes sense. but maybe a good way to pull it off exists
    parser.add_argument('input_filename', type=str,
                        help='File name(s) of the data created by sample_data.py. If more than 1 file, use the format filename1,filename2,filename3 ... '
                             'Do not include file extension')

    # same deal, do I want to include the option of multiple models on 1 plot? I guess if I need to i will
    parser.add_argument('model_filename', type=str,
                        help='File name of the model created by unified_efield.py. Do not include file extension')

    parser.add_argument('-md', '--max_dips',
                        default=None,
                        type=int,
                        help='The maximum number of storm-time events to record and plot. Default is all of them',
                        )

    parser.add_argument('-lr', '--linear_regression_model',
                        default=None,
                        help='File name of a linear regression model to also plot with the neural network. Default is to not plot LR',
                        )

    args = parser.parse_args()

    input_filename = args.input_filename + '.nc'
    model_filename = args.model_filename + '.pth'
    max_dips = args.max_dips
    linear_regression_model_name = args.linear_regression_model

    linear_regression_model = pickle.load(open(linear_regression_model_name, 'rb'))

    layers = model_filename.split('$')[0].split('-')
    values_to_use = model_filename.split('$')[1].split('-')

    NN_layout = np.array([60 * len(values_to_use) + 3])

    NN_layout = np.append(NN_layout, np.array(layers))
    NN_layout = np.append(NN_layout, np.array([3])).astype(int)
    number_of_layers = len(NN_layout) - 2

    NN_dict = {1: NN.NeuralNetwork_1,
               2: NN.NeuralNetwork_2,
               3: NN.NeuralNetwork_3}

    try:
        NeuralNetwork = NN_dict[number_of_layers]
    except:
        raise KeyError("The amount of layers inputted is not available")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    NN_model = NeuralNetwork(NN_layout).to(device)

    NN_model.load_state_dict(torch.load(model_filename))

    data = xr.open_dataset(input_filename)
    dips=get_storm_intervals(data, max_hours=24, max_dips=max_dips)
    plot_dips(dips, data, NN_model, values_to_use, LR_model=linear_regression_model)


if __name__ == '__main__':
    main()