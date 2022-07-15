import torch
from unified_efield_merged import get_inputs
import Neural_Networks as NN
import download_data as dd
import datetime as dt
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import data_manipulation as dm
import argparse

def plot_potential(imef_data, V_data, im_test=None):
    # Note that it is expected that the electric field data is in polar coordinates. Otherwise the potential values are incorrect

    # find L and MLT range used in the data given
    min_Lvalue = imef_data['L'][0, 0].values
    max_Lvalue = imef_data['L'][-1, 0].values
    nL = int(max_Lvalue - min_Lvalue + 1)

    min_MLTvalue = imef_data['MLT'][0, 0].values
    max_MLTvalue = imef_data['MLT'][0, -1].values
    nMLT = int(max_MLTvalue - min_MLTvalue + 1)

    # Create a coordinate grid
    new_values = imef_data['MLT'].values-.5
    phi = (2 * np.pi * new_values / 24).reshape(nL, nMLT)
    r = imef_data['L'].values.reshape(nL, nMLT)-.5

    extra_phi_value = phi[0][0]+2*np.pi

    # The plot comes out missing a section since the coordinates do not completely go around the circle.
    # So we have to copy/paste the first plot point to the end of each of the lists so that the plot is complete
    for counter in range(nL):
        add_to_r = np.append(r[counter], r[counter][0])
        add_to_phi = np.append(phi[0], extra_phi_value)
        add_to_V_data = np.append(V_data[counter], V_data[counter][0])
        if counter==0:
            new_r = [add_to_r]
            new_phi = [add_to_phi]
            new_V_data = [add_to_V_data]
        else:
            new_r = np.append(new_r, [add_to_r], axis=0)
            new_phi= np.append(new_phi, [add_to_phi], axis=0)
            new_V_data = np.append(new_V_data, [add_to_V_data], axis=0)

    # Plot the data
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, subplot_kw=dict(projection='polar'))

    # Plot the electric field
    # Scale makes the arrows smaller/larger. Bigger number = smaller arrows.
    # May need to be changed when more data points are present
    ax1 = axes[0, 0]
    ax1.set_xlabel("Potential")
    ax1.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    # Plot the data. Note that new_V_data is multiplied by -1, since the L/MLT coordinate system has positive x and positive y in the opposite direction as is standard
    im = ax1.contourf(new_phi, new_r, new_V_data*-1, cmap='coolwarm', vmin=-25, vmax=25)
    # plt.clabel(im, inline=True, fontsize=8)
    # plt.imshow(new_V_data, extent=[-40, 12, 0, 10], cmap='RdGy', alpha=0.5)
    fig.colorbar(im, ax=ax1)
    # Draw the earth
    draw_earth(ax1)

    plt.show()


def draw_earth(ax):
    '''
    A handy function for drawing the Earth in a set of Polar Axes
    '''
    ax.fill_between(np.linspace(-np.pi / 2, np.pi / 2, 30), 0, np.ones(30), color='k')
    ax.plot(np.linspace(np.pi / 2, 3 * np.pi / 2, 30), np.ones(30), color='k')


def predict_and_plot(model, time=None, data=None, plot = True, return_pred = False, number_inputs=1):

    # can input either the data you have that corresponds to the time you want to predict (aka the 5 hours of data, with the last input being the time you want to predict)
    # or you can input the time and the data will be downloaded for you
    # if both are given, the data will have priority
    if data is not None:
        complete_data = data
    elif time is not None:
        # we need the data from the 5 hours before the time to the time given
        # But the binned argument requires 1 day of data. so I do this instead
        ti = time - dt.timedelta(hours=5)
        te = time + dt.timedelta(minutes=5)

        mec_data = dd.get_mec_data('mms1', 'srvy', 'l2', ti, te, binned=True)
        kp_data = dd.get_kp_data(ti, te, expand=mec_data['time'].values)
        dst_data = dd.get_dst_data(ti, te, expand=mec_data['time'].values)

        complete_data = xr.merge([mec_data, kp_data, dst_data])
    elif time is None and data is None:
        raise TypeError('Either the desired time or the appropriate data must be given')

    test_inputs = get_inputs(complete_data, remove_nan=False, get_target_data=False)

    base_kp_values = test_inputs[-1].clone()

    for L in range(4, 11):
        for MLT in range(0, 24):
            new_row = base_kp_values.clone()
            new_row[-3] = L
            new_row[-2] = np.cos(np.pi/12*MLT)
            new_row[-1] = np.sin(np.pi/12*MLT)
            even_newer_row = torch.empty((1, 60*number_inputs+3))
            even_newer_row[0] = new_row
            if L == 4 and MLT == 0:
                all_locations = even_newer_row
            else:
                all_locations = torch.cat((all_locations, even_newer_row))

    model.eval()
    x = all_locations
    with torch.no_grad():
        pred = model(x)

    nL = 7
    nMLT = 24

    # Create a coordinate grid
    something = np.arange(0, 24)
    another_thing = np.concatenate((something, something, something, something, something, something, something)).reshape(nL, nMLT)
    phi = (2 * np.pi * another_thing / 24)
    r = np.repeat(np.arange(4, 11), 24).reshape(nL, nMLT)

    Ex_pred = pred[:, 0]
    Ey_pred = pred[:, 1]

    # Start calculating the potential
    L = xr.DataArray(r, dims=['iL', 'iMLT'])
    MLT = xr.DataArray(another_thing, dims=['iL', 'iMLT'])

    # create an empty dataset and insert the predicted cartesian values into it. the time coordinate is nonsensical, but it needs to be time so that rot2polar works
    imef_data = xr.Dataset(coords={'L': L, 'MLT': MLT, 'polar': ['r', 'phi'], 'cartesian': ['x', 'y', 'z']})
    testing_something = xr.DataArray(pred, dims=['time', 'cartesian'], coords={'time': np.arange(0, 168), 'cartesian': ['x', 'y', 'z']})

    pred=pred.reshape(nL, nMLT, 3)

    # Create another dataset containing the locations around the earth as variables instead of dimensions
    imef_data['predicted_efield'] =xr.DataArray(pred, dims=['iL', 'iMLT', 'cartesian'],coords={'L': L, 'MLT': MLT})
    imef_data['R_sc'] = xr.DataArray(np.stack((r, phi), axis=-1).reshape(nL*nMLT, 2), dims=['time', 'polar'], coords={'time': np.arange(0,168), 'polar':['r', 'phi']})

    pred.reshape(nL * nMLT, 3)

    # have to make sure that this actually works correctly. cause otherwise imma be getting some bad stuff
    # Convert the predicted cartesian values to polar
    imef_data['predicted_efield_polar'] = dm.rot2polar(testing_something, imef_data['R_sc'], 'cartesian').assign_coords({'polar': ['r', 'phi']})

    # reshape the predicted polar data to be in terms of L and MLT, and put them into the same dataset
    somethingboi = imef_data['predicted_efield_polar'].values.reshape(nL, nMLT, 2)
    imef_data['predicted_efield_polar_forreal'] = xr.DataArray(somethingboi, dims=['iL', 'iMLT', 'polar'],coords={'L': L, 'MLT': MLT})

    potential = dm.calculate_potential(imef_data, 'predicted_efield_polar_forreal')

    if plot==True:
        # Create figures
        fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, subplot_kw=dict(projection='polar'))
        fig.tight_layout()

        # Plot the electric field
        # Scale makes the arrows smaller/larger. Bigger number = smaller arrows.
        # May need to be changed when more data points are present
        ax1 = axes[0, 0]

        # Note that Ex and Ey are multiplied by -1, since the L/MLT coordinate system has positive x and positive y in the opposite direction as is standard
        ax1.quiver(phi, r, -1 * Ex_pred, -1 * Ey_pred, scale=5)
        ax1.set_xlabel("Electric Field")
        ax1.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
        ax1.set_theta_direction(1)

        # Draw the earth
        draw_earth(ax1)

        plot_potential(imef_data, potential)

    if return_pred == True:
        return imef_data, potential


def main():
    parser = argparse.ArgumentParser(
        description='PUT DESCRIPTION HERE'
    )

    parser.add_argument('model_filename', type=str,
                        help='Name of the file containing the trained NN. Do not include file extension')

    parser.add_argument('time_to_predict', type=str,
                        help='The time that the user wants predict the electric field and electric potential for')

    args = parser.parse_args()

    model_filename = args.model_filename+'.pth'

    layers = model_filename.split('$')[0].split('-')
    values_to_use = model_filename.split('$')[1].split('-')

    # change this to be 183 when symh is involved
    if values_to_use[0] == 'All':
        NN_layout = np.array([123])
    else:
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

    time = dt.datetime.strptime(args.time_to_predict, '%Y-%m-%dT%H:%M:%S')

    predict_and_plot(model, time=time, number_inputs=len(values_to_use))


if __name__ == '__main__':
    main()