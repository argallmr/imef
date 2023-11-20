import numpy as np
import Neural_Networks as NN
import torch
import pandas as pd
# from torch.utils.data import DataLoader, TensorDataset


def get_predictor_counts(number_of_inputs):
    # return 60*number_of_inputs+3
    return 60*number_of_inputs+5


# given a list of layers, return a Neural network class.
# Note that this can be expanded on dramatically, including but not limited to more layers and different types of NNs.
# Currently 1, 2, and 3 are all linear layers only. If a different type of NN is needed, this will have to be changed somehow
def get_NN(NN_layout, device='cpu'):
    NN_dict = {1: NN.NeuralNetwork_1,
               2: NN.NeuralNetwork_2,
               3: NN.NeuralNetwork_3}

    try:
        NeuralNetwork = NN_dict[len(NN_layout)-2]
    except:
        raise KeyError("The amount of layers inputted is not available")

    model = NeuralNetwork(NN_layout).to(device)

    return model


def output_error(values_to_use, parameters, number_of_layers, final_test_error, file_to_output_to = 'test_errors.txt'):
    # This function outputs the results of kfold cross validation from create_neural_network to a text file

    counter = 0
    values_string = ''
    for value in values_to_use:
        values_string += value
    string = str('Inputs: ' + values_string + ' || Layers: ')
    for parameter in parameters:
        if counter % 2 == 0 and counter < 2 * number_of_layers - 2:
            string = string + str(len(parameter)) + '-'
        elif counter % 2 == 0 and counter == 2 * number_of_layers - 2:
            string = string + str(len(parameter))
        counter += 1

    # Output the properties and the test results of the NN to a file called test_errors.txt
    put_error_here = open(file_to_output_to, 'a')
    output = string + str(
        ' || ExMSE: ' + str(final_test_error[0]) + ' || EyMSE: ' + str(final_test_error[1]) + ' || EzMSE: '
        + str(final_test_error[2]) + ' || Total E MSE: ' + str(np.sum(final_test_error)) + '\n')
    put_error_here.write(output)

    return output


def get_NN_inputs(imef_data, remove_nan=True, get_target_data=True, use_values=['Kp'], usetorch=True, undersample=None):
    # This could be made way more efficient if I were to make the function not store all the data even if it isn't used. But for sake of understandability (which this has little of anyways)
    # the random_undersampling argument should be a float, if it is not none. The float represents the quiet_storm_ratio

    if 'Kp' in use_values:
        Kp_data = imef_data['Kp']
    if 'Dst' in use_values:
        Dst_data = imef_data['Dst']
    if 'Symh' in use_values:
        Symh_data = imef_data['Sym-H']

    if remove_nan == True:
        imef_data = imef_data.where(np.isnan(imef_data['E_con'][:, 0]) == False, drop=True)
    if undersample != None and len(imef_data['time'].values) != 0:
        imef_data = random_undersampling(imef_data, quiet_storm_ratio=undersample)

    # Note that the first 5 hours of data cannot be used, since we do not have the preceding 5 hours of index data to get all the required data. Remove those 5 hours
    # Since there is now a try in the for loop, this shouldn't be needed. But just in case something strange comes up ill leave it here
    # imef_data = imef_data.where(imef_data['time']>=(imef_data['time'].values[0]+np.timedelta64(5, 'h')), drop=True)

    design_matrix_array=None
    if get_target_data == True:
        times_to_keep = []
    for counter in range(0, len(imef_data['time'].values)):
        new_data_line = []
        time_intervals = pd.date_range(end=imef_data['time'].values[counter], freq='5T', periods=60)
        try:
            if 'Kp' in use_values:
                Kp_index_data = Kp_data.sel(time=time_intervals).values.tolist()
                new_data_line += Kp_index_data
            if 'Dst' in use_values:
                Dst_index_data = Dst_data.sel(time=time_intervals).values.tolist()
                new_data_line += Dst_index_data
            if 'Symh' in use_values:
                Symh_index_data = Symh_data.sel(time=time_intervals).values.tolist()
                new_data_line += Symh_index_data

            # Along with the indices, we include 3 extra values to train on: The distance from the Earth (L), cos(MLT), and sin(MLT)
            # the_rest_of_the_data = np.array([imef_data['L'].values[counter], np.cos(np.pi / 12 * imef_data['MLT'].values[counter]), np.sin(np.pi / 12 * imef_data['MLT'].values[counter])]).tolist()

            # Along with the indices, we include 5 extra values to train on: The distance from the Earth (L), cos(MLT), sin(MLT), cos(MLAT), and sin(MLAT)
            the_rest_of_the_data = np.array([imef_data['L'].values[counter], np.cos(np.pi / 12 * imef_data['MLT'].values[counter]), np.sin(np.pi / 12 * imef_data['MLT'].values[counter]),
                                             np.cos(imef_data['MLAT'].values[counter]),np.sin(imef_data['MLAT'].values[counter])]).tolist()
            new_data_line += the_rest_of_the_data

            if design_matrix_array==None:
                design_matrix_array = [new_data_line]
            else:
                design_matrix_array.append(new_data_line)

            if get_target_data==True:
                times_to_keep.append(imef_data['time'].values[counter])
        except Exception as ex:
            # This should only be entered when there is not enough data to fully create the NN inputs required (aka previous 5 hours of index data, and location data)
            # print(ex)
            # raise ex
            pass

    if usetorch==True:
        design_matrix_array = torch.tensor(design_matrix_array)
    else:
        design_matrix_array = np.array(design_matrix_array)

    if get_target_data == True:
        efield_data = imef_data['E_con'].sel(time=times_to_keep).values

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


# def train_NN_kfold():
#
#
# def train_NN():


def random_undersampling(data, threshold=-40, quiet_storm_ratio=1.0):
    # Tbh not quite sure if I understood the algorithm correctly. It does what I think it wants, so I'll stick with it for now.
    # To be clear, this definitely undersamples the data. However the way I do it is by reducing the number of datapoints in each bin by a certain percentage. The authors may not do the same
    # But here is the link in case it's needed: https://link.springer.com/article/10.1007/s41060-017-0044-3

    intervals_of_storm_data = get_storm_intervals(data, threshold=threshold, max_hours=0)
    storm_counts=0
    for start, end in intervals_of_storm_data:
        storm_counts += end-start
    if storm_counts == 0:
        print(Warning('There is no storm data in the given dataset. Returning given data.'))
        return data
    quiet_counts=len(data['time'])-storm_counts

    if quiet_counts > storm_counts:
        bins_to_undersample = reverse_bins(intervals_of_storm_data, len(data['time']))
        percent_to_reduce = quiet_storm_ratio * storm_counts / quiet_counts

        if percent_to_reduce >= 1:
            raise ValueError('quiet_storm_ratio is too large. The max value for this dataset is '+str(quiet_counts/storm_counts))
        elif percent_to_reduce <= 0:
            raise ValueError('quiet_storm ratio is too small. It must be greater than 0')

        all_times = []
        for start, end in intervals_of_storm_data:
            all_times.append(data['time'].values[start:end])
        for start, end in bins_to_undersample:
            new_times_in_bin = np.random.choice(data['time'][start:end], int(percent_to_reduce * (end - start)), replace=False)
            all_times.append(new_times_in_bin)
        all_times = np.concatenate(all_times)
        all_times = np.sort(all_times)

        undersampled_data = data.sel(time=all_times)

        return undersampled_data
    else:
        # I don't know if a) this will ever come up, or b) if this did come up, we would want to undersample the storm data. So raising error for now
        print(Warning('There is more storm data than quiet data. Skipping undersampling.'))
        return data


# just realized this could probably be made significantly easier with binned_statistic. oh well
def get_storm_intervals(data, mode='Dst', threshold=-40, max_hours=24, max_dips=None):
    # given a datafile from sample_data.py, this function returns a list of intervals that represent the storm data
    # dips has the first and last indices that contain all the data
    # max_hours is the cutoff point: after x (default=24) hours of data being below the threshold, save the dip locations
    # max_dips is return the first int amount of dips found in the data. Default is to return all of them

    timescale = data['time'].values[1] - data['time'].values[0]
    if mode == 'Dst':
        bins_per_timescale = int(np.timedelta64(1, 'h')/timescale)
    elif mode == 'Sym-H':
        bins_per_timescale = int(np.timedelta64(1, 'm')/timescale)
    else:
        raise ValueError('Mode must be either Dst or Sym-H')

    dips=[]
    counter=0
    number_of_dips=0
    while counter < len(data[mode].values):
        if number_of_dips == max_dips:
            break
        if data[mode].values[counter] <=threshold:
            loop=True
            another_counter=counter
            hours_under_thresh=0
            while loop==True:
                if max_hours == None or max_hours==0:
                    if another_counter >= len(data[mode].values) or data[mode].values[another_counter] > threshold:
                        loop=False
                        dips.append([counter, another_counter])
                        counter=another_counter
                    another_counter += bins_per_timescale
                    if another_counter > len(data[mode].values):
                        another_counter = len(data[mode].values)
                else:
                    if data[mode].values[another_counter] > threshold:
                        hours_under_thresh+=1
                    else:
                        hours_under_thresh=0
                    another_counter += bins_per_timescale
                    if hours_under_thresh==max_hours or another_counter >= len(data[mode].values): # if there is int(max_hours) hours without a dip over (threshold) nT, then stop and record the index
                        loop=False
                        if another_counter > len(data[mode].values):
                            another_counter = len(data[mode].values)
                        if counter-bins_per_timescale*max_hours<0:
                            dips.append([0, another_counter])
                        else:
                            dips.append([counter-bins_per_timescale*max_hours, another_counter]) # record 1 day of data before the initial drop, and all the data until 1 day after
                        counter=another_counter
                        number_of_dips+=1

        counter+=1

    return dips


def reverse_bins(bins, length_of_data, keep_overlapping = False):
    reversed_bins = []
    counter = 0

    while counter < len(bins):
        if counter == 0:
            # put first bin in list (if needed)
            if bins[0][0] == 0:
                pass
            else:
                reversed_bins.append([0, bins[0][0]])
        else:
            # Put reversed bins in list. if the bins from bins overlap with each other (which can occur in get_storm_intervals), then ignore them while reversing unless specified otherwise
            if keep_overlapping == True or bins[counter - 1][1] < bins[counter][0]:
                reversed_bins.append([bins[counter - 1][1], bins[counter][0]])
        counter += 1

    # put last bin in list (if needed)
    if bins[-1][1] != length_of_data:
        reversed_bins.append([bins[-1][1], length_of_data])

    return reversed_bins