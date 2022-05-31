import numpy as np
import xarray as xr
import argparse
import torch
import matplotlib.pyplot as plt
import datetime as dt
from pandas.plotting import register_matplotlib_converters
import Neural_Networks as NN
from unified_efield import get_inputs
from sklearn.linear_model import LinearRegression

# something messed up here. double check on desktop
def test_linear_regression(dip, data):
    values_to_use = 'All'
    total_inputs, total_targets = get_inputs(data, use_values = values_to_use, usetorch=False)
    train_inputs = np.append(total_inputs[0:dip[0]], total_inputs[dip[1]:len(total_inputs)-1])
    train_targets = np.append(total_targets[0:dip[0]], total_targets[dip[1]:len(total_targets)-1])
    test_inputs = total_inputs[dip[0]:dip[1]]
    test_targets = total_targets[dip[0]:dip[1]]
    LR = LinearRegression()
    LR.fit(train_inputs, train_targets)

    return LR

# put the plotting function into plot_nc_data when I move this to desktop
# USE SYM-H, swap out when I get a file that has sym-h. Or should I?
def plot_dips(dips, data):
    # plot everything, then plot the dips
    # for some reason the time version has a couple messed up areas. I don't think this is my fault, rather matplotlibs.
    # Hopefully I just dont use this bit anyways and it doesnt matter
    # plt.plot(data['time'].values, data['DST'].values)
    # plt.show()
    # plt.plot(np.linspace(0, len(data['DST'].values), len(data['DST'].values)), data['DST'].values)
    # plt.show()
    for start, end in dips:
        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
        fig.tight_layout()
        dst_data = data['DST'].values[start:end]
        # dunno what this is, but pandas said to do it so here it is
        register_matplotlib_converters()
        ax1=axes[0,0]
        ax1.set_ylim([-150, 50])
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Dst')
        ax1.plot(data['time'].values[start:end], dst_data)

        # repeat for y and z if needed. Or just take the magnitude and plot that. Or make that a 4th plot
        ax2=axes[0,1]
        E_data = np.sqrt(data['E_GSE'].values[start:end,0]**2 + data['E_GSE'].values[start:end,1]**2 + data['E_GSE'].values[start:end,0]**2)
        ax2.plot(data['time'].values[start:end], E_data)
        ax2.set_ylim([-3, 3])
        ax2.set_xlabel('Date')
        ax2.set_ylabel('||E||')

        # make a new one where I can put in the data and the indices. Maybe call it test_linear_regression or something. Maybe rename linear_regression to
        # get_test_error_LR or something
        # lin_reg = test_linear_regression([start,end], data)
        plt.show()


# Do I want to use mode or no?
def get_x_dips(data, x=None, mode='dst'):
    # not efficient, but works well enough. Maybe do a random guess or something at some point for 10 diff point every time I run?
    # dips has the first and last indices that contain all the data I want to plot
    dips=[]
    counter=0
    while counter < len(data['DST'].values):
        if data['DST'].values[counter] <=-50:
            loop=True
            another_counter=counter
            hours_under_50=0
            while loop==True:
                if data['DST'].values[another_counter] > -50:
                    hours_under_50+=1
                else:
                    hours_under_50=0
                another_counter += 12

                if hours_under_50==24: # if there is 24 hours without a dip over 50 nT, then stop and record the index
                    loop=False
                    dips.append([counter-288, another_counter]) # record 1 day of data before the initial drop, and all the data until 1 day after
                    counter=another_counter

            # this stops it at 10. Can remove this to get all of them. There were 35 with 1 day before and after. Maybe randomly pick after getting all of them?
            if x is not None:
                if len(dips)==x:
                    break

        counter+=1

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
    # parser.add_argument('model_filename', type=str,
    #                     help='File name of the model created by unified_efield.py. If more than 1 model, use the format filename1,filename2,filename3 ... '
    #                          'Do not include file extension')

    # Maybe introduce argument that is a list of times to use instead of the first 10 dst drops. Or a number of drops and use the x argument

    args = parser.parse_args()
    print(1)

    input_filename = args.input_filename + '.nc'
    print(2)

    data = xr.open_dataset(input_filename)
    print(3)

    LR = test_linear_regression([20, 100], data)

    print(LR)

    dips=get_x_dips(data)

    plot_dips(dips, data)

    # Either plot the NN and LR in plot_dips, or move the for loop out of plot_dips and out here, and run do plot_dips and plot_NN/LR once per loop




if __name__ == '__main__':
    main()