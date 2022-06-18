import numpy as np
import gif
from predict_unified_efield import predict_and_plot
import torch
import argparse
import Neural_Networks as NN
import datetime as dt
import matplotlib.pyplot as plt
import plot_nc_data as xrplot

def draw_earth(ax):
    '''
    A handy function for drawing the Earth in a set of Polar Axes
    '''
    ax.fill_between(np.linspace(-np.pi / 2, np.pi / 2, 30), 0, np.ones(30), color='k')
    ax.plot(np.linspace(np.pi / 2, 3 * np.pi / 2, 30), np.ones(30), color='k')

@gif.frame
def plot_potential_for_gif(imef_data, V_data, im_test=None):
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
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, subplot_kw=dict(projection='polar'))

    # Plot the electric field
    # Scale makes the arrows smaller/larger. Bigger number = smaller arrows.
    # May need to be changed when more data points are present
    ax1 = axes[0, 1]
    ax1.set_xlabel("Potential")
    ax1.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    # Plot the data. Note that new_V_data is multiplied by -1, since the L/MLT coordinate system has positive x and positive y in the opposite direction as is standard
    im = ax1.contourf(new_phi, new_r, new_V_data*-1, cmap='coolwarm', vmin=-25, vmax=25)
    # plt.clabel(im, inline=True, fontsize=8)
    # plt.imshow(new_V_data, extent=[-40, 12, 0, 10], cmap='RdGy', alpha=0.5)
    if im_test is not None:
        fig.colorbar(im_test, ax=ax1)
    else:
        fig.colorbar(im, ax=ax1)
    # Draw the earth
    draw_earth(ax1)

    ax2 = axes[0, 0]

    plotted_variable = 'predicted_efield'
    Ex = imef_data[plotted_variable].loc[:, :, 'x'].values.reshape(nL, nMLT)
    Ey = imef_data[plotted_variable].loc[:, :, 'y'].values.reshape(nL, nMLT)

    # Note that Ex and Ey are multiplied by -1, since the L/MLT coordinate system has positive x and positive y in the opposite direction as is standard
    ax2.quiver(phi, r, -1 * Ex, -1 * Ey, scale=14)
    ax2.set_xlabel("Electric Field")
    ax2.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    ax2.set_theta_direction(1)

    # Draw the earth
    draw_earth(ax2)


def get_good_colorplot(imef_data, V_data):

    min_Lvalue = imef_data['L'][0, 0].values
    max_Lvalue = imef_data['L'][-1, 0].values
    nL = int(max_Lvalue - min_Lvalue + 1)

    min_MLTvalue = imef_data['MLT'][0, 0].values
    max_MLTvalue = imef_data['MLT'][0, -1].values
    nMLT = int(max_MLTvalue - min_MLTvalue + 1)

    # Create a coordinate grid
    new_values = imef_data['MLT'].values - .5
    phi = (2 * np.pi * new_values / 24).reshape(nL, nMLT)
    r = imef_data['L'].values.reshape(nL, nMLT) - .5

    extra_phi_value = phi[0][0] + 2 * np.pi

    # The plot comes out missing a section since the coordinates do not completely go around the circle.
    # So we have to copy/paste the first plot point to the end of each of the lists so that the plot is complete
    sign=1
    for counter in range(nL):
        add_to_r = np.append(r[counter], r[counter][0])
        add_to_phi = np.append(phi[0], extra_phi_value)
        add_to_V_data = np.append(V_data[counter], sign*25)
        if counter == 0:
            new_r = [add_to_r]
            new_phi = [add_to_phi]
            new_V_data = [add_to_V_data]
        else:
            new_r = np.append(new_r, [add_to_r], axis=0)
            new_phi = np.append(new_phi, [add_to_phi], axis=0)
            new_V_data = np.append(new_V_data, [add_to_V_data], axis=0)
        sign=sign*-1

    # Plot the data
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, subplot_kw=dict(projection='polar'))

    # Plot the electric field
    # Scale makes the arrows smaller/larger. Bigger number = smaller arrows.
    # May need to be changed when more data points are present
    ax1 = axes[0, 0]
    ax1.set_xlabel("Potential")
    ax1.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    # Plot the data. Note that new_V_data is multiplied by -1, since the L/MLT coordinate system has positive x and positive y in the opposite direction as is standard
    im = ax1.contourf(new_phi, new_r, new_V_data * -1, cmap='coolwarm', vmin=-25, vmax=25)

    return im


def main():
    parser = argparse.ArgumentParser(
        description='PUT DESCRIPTION HERE'
    )

    parser.add_argument('model_filename', type=str,
                        help='Name of the file containing the trained NN. Do not include file extension')

    parser.add_argument('start_time', type=str,
                        help='The start time that the user wants to create the electric potential gif forL "YYYY-MM-DDTHH:MM:SS"')

    parser.add_argument('end_time', type=str,
                        help='The end time that the user wants to create the electric potential gif for: "YYYY-MM-DDTHH:MM:SS"')

    parser.add_argument('time_interval', type=str,
                        help='How often the potential should be calculated and plotted over the given time range, in hours. Decimals are allowed')

    parser.add_argument('fig_filename', type=str,
                        help='Desired name of the file containing the newly created gif. Do not include file extension')

    args = parser.parse_args()

    model_filename = args.model_filename + '.pth'

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

    start_time = dt.datetime.strptime(args.start_time, '%Y-%m-%dT%H:%M:%S')
    end_time = dt.datetime.strptime(args.end_time, '%Y-%m-%dT%H:%M:%S')

    time_change = dt.timedelta(hours=int(args.time_interval))

    ti = start_time

    gif.options.matplotlib["dpi"] = 300
    frames = []

    duration=0

    while ti <= end_time:
        # could be made slightly more accurate since predict and plot keeps downloading stuff over and over. Maybe download ti to te outside of predict and plot and use data arg?
        imef_data, potential = predict_and_plot(model, ti, return_pred=True, plot=False)
        # this is for getting 1 good colorplot to use for every frame of the gif. Since the min and max V value is set to +-25 in plot_potential_for_gif, the colormap is the same in each frame
        # note this doesn't need to be run every time. I could run the first iteration of predict_and_plot outside the loop, run this once, then do while loop, but not really that big a deal
        im = get_good_colorplot(imef_data, potential)
        frames.append(plot_potential_for_gif(imef_data, potential, im))
        ti += time_change
        duration += 1.5

    #create gif here
    gif.save(frames, args.fig_filename+'.gif',
             duration=duration, unit="s",
             between="startend")


if __name__ == '__main__':
    main()