import numpy as np
import xarray as xr
import argparse
import torch
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import imef.efield.model_creation.Neural_Networks as NN
from imef.data.data_manipulation import get_NN_inputs, get_storm_intervals
from sklearn.linear_model import LinearRegression

def pred_w_NN(dip, data, model, values_to_use):
    # would definitely be more efficient to slice data first, then put into get_inputs, but where is being a pain (it has the error that sample_data has been getting, I can't figure it out)
    test_inputs = get_NN_inputs(data, use_values=values_to_use, remove_nan=False, get_target_data=False)
    test_inputs = test_inputs[dip[0]:dip[1]]
    model.eval()
    x = test_inputs
    with torch.no_grad():
        pred = model(x)

    return pred


def pred_w_linear_regression(dip, data, values_to_use):
    # definitely not the most efficient way to do this, but it's what I came up with on short notice. Could come back to this is I have extra time
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


def plot_dips(dips, data, model, values_to_use, mode='Dst'):
    for start, end in dips:
        print('PLOTTING')
        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
        fig.tight_layout()
        # dunno what this is, but pandas said to do it so here it is
        register_matplotlib_converters()
        ax1=axes[0,0]
        ax1.set_ylim([-150, 50])
        ax1.set_xlabel('Date')
        ax1.set_ylabel(mode)
        ax1.plot(data['time'].values[start:end], data[mode].values[start:end])

        my_xticks = ax1.get_xticks()
        ax1.set_xticks([my_xticks[0], my_xticks[-1]])

        # repeat for y and z if needed. Or just take the magnitude and plot that. Or make that a 4th plot
        ax2=axes[0,1]
        E_data = np.sqrt(data['E_con'].values[start:end,0]**2 + data['E_con'].values[start:end,1]**2 + data['E_con'].values[start:end,0]**2)
        ax2.plot(data['time'].values[start:end], E_data, label='MMS Data')
        ax2.set_ylim([0, 10])
        ax2.set_xlabel('Date')
        ax2.set_ylabel('||E||')

        # predict lin_reg in here, and return the predicted values.
        LR_predicted = pred_w_linear_regression([start,end], data, values_to_use)

        ax2.plot(data['time'].values[start:end], np.sqrt(LR_predicted[:,0]**2 + LR_predicted[:,1]**2 + LR_predicted[:,2]**2), label='Linear Regression')

        NN_predicted = pred_w_NN([start,end], data, model, values_to_use)

        ax2.plot(data['time'].values[start:end],np.sqrt(NN_predicted[:, 0] ** 2 + NN_predicted[:, 1] ** 2 + NN_predicted[:, 2] ** 2),label='Neural Network')

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

    # Maybe introduce argument that is a list of times to use instead of the first 10 dst drops. Or a number of drops and use the x argument

    args = parser.parse_args()

    input_filename = args.input_filename + '.nc'
    model_filename = args.model_filename + '.pth'

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
    model = NeuralNetwork(NN_layout).to(device)

    model.load_state_dict(torch.load(model_filename))

    data = xr.open_dataset(input_filename)

    dips=get_storm_intervals(data, max_hours=24)
    print('DONE ALL')
    plot_dips(dips, data, model, values_to_use)


if __name__ == '__main__':
    main()