import numpy as np
from matplotlib import pyplot as plt

# For debugging purposes
np.set_printoptions(threshold=np.inf)

# The expectation of all the functions in this program is that the data inputs are all xarray objects, created from either sample_data.py, or store_edi_data.py
# This is important because the variables automatically choose the names of the variables that are outputted from those programs.

# fig.savefig('testing.pdf', format='pdf')
# This is how to save a plot as a pdf

# Important note: the L/MLT coordinate system has positive x facing left and positive y facing down, the opposite of the typical x-y system.
# There will be a large number of -1*(data) in these functions as a result

# TODO: I should probably make a choice for consistency: either make the person input the variable, make it an optional argument, or don't implement anything else
#  Probably remove the plotting from store_efield_data
#  I might have to generalize some things to make it not Kp only (specifically if I start doing IEF stuff)

def draw_earth(ax):
    '''
    A handy function for drawing the Earth in a set of Polar Axes
    '''
    ax.fill_between(np.linspace(-np.pi / 2, np.pi / 2, 30), 0, np.ones(30), color='k')
    ax.plot(np.linspace(np.pi / 2, 3 * np.pi / 2, 30), np.ones(30), color='k')


def plot_efield(imef_data, plotted_variable, mode='cartesian', log_counts=False):

    # find L and MLT range used in the data given
    min_Lvalue = imef_data['L'][0, 0].values
    max_Lvalue = imef_data['L'][-1, 0].values
    nL = int(max_Lvalue - min_Lvalue + 1)

    min_MLTvalue = imef_data['MLT'][0, 0].values
    max_MLTvalue = imef_data['MLT'][0, -1].values
    nMLT = int(max_MLTvalue - min_MLTvalue + 1)

    # Create a coordinate grid
    phi = (2 * np.pi * imef_data['MLT'].values / 24).reshape(nL, nMLT)
    r = imef_data['L'].values.reshape(nL, nMLT)
    if mode=='polar':
        Er = imef_data[plotted_variable].loc[:, :, 'r'].values.reshape(nL, nMLT)
        Ephi = imef_data[plotted_variable].loc[:, :, 'phi'].values.reshape(nL, nMLT)

        # Convert to cartesian coordinates
        # Scaling the vectors in the plotting doesn't work correctly unless this is done.
        Ex = Er * np.cos(phi) - Ephi * np.sin(phi)
        Ey = Er * np.sin(phi) + Ephi * np.cos(phi)
    elif mode=='cartesian':
        Ex = imef_data[plotted_variable].loc[:, :, 'x'].values.reshape(nL, nMLT)
        Ey = imef_data[plotted_variable].loc[:, :, 'y'].values.reshape(nL, nMLT)

    # Create figures
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, subplot_kw=dict(projection='polar'))
    fig.tight_layout()

    # Plot the electric field
    # Scale makes the arrows smaller/larger. Bigger number = smaller arrows.
    # May need to be changed when more data points are present
    ax1 = axes[0, 0]

    # Note that Ex and Ey are multiplied by -1, since the L/MLT coordinate system has positive x and positive y in the opposite direction as is standard
    ax1.quiver(phi, r, -1 * Ex, -1 * Ey, scale=14)
    ax1.set_xlabel("Electric Field")
    ax1.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    ax1.set_theta_direction(1)

    # Draw the earth
    draw_earth(ax1)

    # Plot the number of data points in each bin
    ax2 = axes[0, 1]
    ax2.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    ax2.set_xlabel("Count")
    # Create the name of the counts variable associated with the plotted variable inputted
    counts_name = plotted_variable[:len(plotted_variable)-4]+'count'

    if log_counts == True:
        im = ax2.pcolormesh(phi, r, np.log10(imef_data[counts_name].data), cmap='YlOrRd', shading='auto')
    else:
        im = ax2.pcolormesh(phi, r, imef_data[counts_name].data, cmap='YlOrRd', shading='auto')
    fig.colorbar(im, ax=ax2)
    draw_earth(ax2)


def plot_potential(nL, nMLT, imef_data, V_data):
    # Note that it is expected that the electric field data is in polar coordinates. Otherwise the potential values are incorrect

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


def line_plot(data, mode='cartesian', MLT_range=None, Kp_range=None):
    # Note that this function only works with Kp bins of size 1. A generalized version could be created, but the combined 7-8 and 8-9 line would have to be removed

    # If I can find a good way to do so, cut down on if statement and for loops. I am 4 indents in in some places
    # Also, I'm not sure if polar actually works. I'm not sure what would go wrong just from looking at it, but I haven't tested it. So that could be a problem
    # This doesn't work for ranges that cross over 0 MLT (eg. 21 MLT to 3 MLT)

    # This is the amount of graphs that this function will create. It depends on the number of coordinates (1 plot per coordinate). Polar has 2 (r, \theta), cartesian has 3 (x, y, z)
    if mode == 'cartesian':
        rounds=3
    elif mode == 'polar':
        rounds=2
    else:
        raise Exception('the mode should either be cartesian or polar. Default is cartesian')

    # Set up some initial values. Mostly self-explanatory
    min_Lvalue = data['L'][0, 0].values
    max_Lvalue = data['L'][-1, 0].values
    nL = int(max_Lvalue - min_Lvalue + 1)

    # the MLT_range argument is for when you want to plot only a specific range of MLT values. Default is all MLT values (0 to 24)
    if MLT_range is None:
        MLT_range = [0, 24]

    # the Kp_range argument is for when you want to plot only a specific range of Kp values. Default is all Kp values (0 to 9)
    if Kp_range is None:
        # Note that while Kp goes from 0 to 9, since the last 2 bins are supposed to be combined, I use 8 instead of 9. It should not go higher than 8 or lower than 0
        Kp_range = [0, 8]

    # Designate step size
    step = float(1)

    # Create the figure for the plots to be made on
    fig, axes = plt.subplots(nrows=1, ncols=rounds, squeeze=False)
    fig.tight_layout()


    # Create lists of the colors and labels used in the plot
    colors = ['purple', 'mediumpurple', 'cornflowerblue', 'lightseagreen', 'forestgreen', 'yellow', 'orange', 'red']
    labels = ['[0,1)', '[1,2)', '[2,3)', '[3,4)', '[4,5)', '[5,6)', '[6,7)', '[7,9]']
    if mode=="cartesian":
        axes_labels = [['L', 'Ex (GSE)'], ['L', 'Ey (GSE)'], ['L', 'Ez (GSE)']]
    elif mode == 'polar':
        axes_labels = [['L', 'Er (GSE)'], ['L', 'Etheta (GSE)']] #fix theta at some point

    # The line plot for just Efield vs L
    # This process is repeated on 2 or 3 separate plots, one for the each component (x, y, z) or (r, \theta)
    for component in range(rounds):
        # Since the way the variables are named is weird (mainly that they are 0 to 1.0, 1.0 to 2.0, etc. 0 is messed up),
        # I'm going to leave this here, so that the name creation works as it should. That being said, just pretend that counter2 is Kp_value, because it basically is
        counter2 = 0

        # There are 2 (polar) or 3 (cartesian) plots, one for each component. Select the right plot here
        ax2 = axes[0, component]

        # Choose x axis bounds
        ax2.set_xlim([min_Lvalue, max_Lvalue])
        ax2.set_ylim([-.2, 1.8])

        # Label the axes
        ax2.set_xlabel(axes_labels[component][0])
        ax2.set_ylabel(axes_labels[component][1])

        # Iterate over each Kp value. Kp ranges from 0 to 9. Since this function combines 7-8 and 8-9, the highest possible Kp_value number is 7
        for Kp_value in range(Kp_range[0], Kp_range[1]):
            # Where the values for each line will be stored
            list = []

            # Designate the name of the variable being plotted. Eg. E_GSE_Kp_1.0_to_2.0 when Kp_value is 1
            if mode == 'polar':
                string = 'E_GSE_polar_Kp_' + str(counter2) + '_to_' + str(counter2 + step)
            else:
                string = 'E_GSE_Kp_' + str(counter2) + '_to_' + str(counter2 + step)

            # Since the values of Kp=7 to 8 and Kp=8 to 9 are supposed to be put together, a special case is made.
            if Kp_value == 7:
                # Make the name for the Kp value of 8-9 (7-8 was already created in previous lines)
                if mode == 'polar':
                    string2 = 'E_GSE_polar_Kp_' + str(counter2 + step) + '_to_' + str(counter2 + 2 * step)
                else:
                    string2 = 'E_GSE_Kp_' + str(counter2 + step) + '_to_' + str(counter2 + 2 * step)

                for counter3 in range(nL):
                    # Get the sum of the mean and count values for the electric field
                    variable_mean = data[string + '_mean'][counter3, MLT_range[0]:MLT_range[1], component].values + data[string2 + '_mean'][counter3, MLT_range[0]:MLT_range[1],component].values
                    variable_count = data[string + '_count'][counter3, MLT_range[0]:MLT_range[1]].values + data[string2 + '_count'][counter3, MLT_range[0]:MLT_range[1]].values

                    # Calculate the weighted average of the electric field
                    weighted_values = weighted_average(variable_mean, variable_count)

                    # Store the average value of the electric field over both Kp values at each L value in list
                    list.append(weighted_values)

                    # At every point put into the list, put a little dot (just for making plot look nice)
                    ax2.plot(data['iL'].values[counter3] + 4.5, weighted_values, color=colors[Kp_value], marker="o", markersize=3)

                # plot the values put into the list
                ax2.plot(data['iL'].values + 4.5, list, color=colors[Kp_value], label=labels[Kp_value])

            # For any time the Kp value is below
            else:
                # Iterate through each L value
                for counter3 in range(nL):
                    # Get the sum of the mean and count values for the electric field
                    variable_mean = data[string + '_mean'][counter3, MLT_range[0]:MLT_range[1], component].values
                    variable_count = data[string + '_count'][counter3, MLT_range[0]:MLT_range[1]].values

                    # Calculate the weighted average of the electric field
                    weighted_values = weighted_average(variable_mean, variable_count)

                    # Store the average value of the electric field at each L value in list
                    list.append(weighted_values)

                    # At every point put into the list, put a little dot (just for making plot look nice)
                    ax2.plot(data['iL'].values[counter3] + 4.5, weighted_values, color=colors[Kp_value], marker="o", markersize=3)

                # plot the values put into the list
                ax2.plot(data['iL'].values + 4.5, list, color=colors[Kp_value], label=labels[Kp_value])

            # Increase the counter for the next loop
            counter2 += step

        # Create a legend, with labels created in the Kp loop
        if component == 0:
            fig.legend(title="Kp Index")

    plt.show()


# This is for the line_plot function. I could move this to data_manipulation if I want (it does make more sense in there),
# but it isn't used anywhere else. If I do end up using this in other places I'll move it to data_manipulation
# def weighted_average(data, data_counts):
#     total = sum(data * data_counts)
#     total_counts = sum(data_counts)
#     # Idk what else to do when I have 0 counts. If this isn't done then we get nan and no line printed
#     if total_counts == 0:
#         return 0
#     else:
#         return total/total_counts

def weighted_average(data, data_counts):
    total = sum(data)
    total_counts = len(data)
    # Idk what else to do when I have 0 counts. If this isn't done then we get nan and no line printed
    if total_counts == 0:
        return 0
    else:
        return total/total_counts


def counts_bar_plot(data, color='blue', Kp_range=None, log=True):
    # This is a single bar graph, with counts on the y axis and Kp value on the x axis
    # This only works with step size of Kp bins being 1

    # Create the figure
    fig_counts, axes_counts = plt.subplots(nrows=1, ncols=1, squeeze=False)

    # Where the counts data is stored
    list = []

    # The argument Kp_range is a list that has two values, the lowest and highest Kp values to be plotted. Kp ranges from 0 to 9, and that will be the default
    if Kp_range is None:
        Kp_range = [0, 9]

    # Iterate through every Kp value
    for Kp_values in range(Kp_range[0], Kp_range[1]):
        # Create the name of the variable that has the data in this Kp range
        if Kp_values == 0:
            string = 'E_GSE_Kp_' + str(Kp_values) + '_to_' + str(Kp_values + 1) + '.0_count'
        else:
            string = 'E_GSE_Kp_' + str(Kp_values) + '.0_to_' + str(Kp_values + 1) + '.0_count'

        # Take the counts data and add it to the list
        list.append(data[string].values.sum())

    # Adding axes labels

    # Plot the data. the log argument allows the user to change the count values into log10(counts), for easier viewing in some cases
    if log==False:
        axes_counts[0][0].bar(np.arange(Kp_range[0] + .5, Kp_range[1] + .5, 1), list, color=color)

        # Adding axes labels
        axes_counts[0][0].set_xlabel('Kp_value')
        axes_counts[0][0].set_ylabel('Counts')
        axes_counts[0][0].set_title('Counts')
    else:
        axes_counts[0][0].bar(np.arange(Kp_range[0] + .5, Kp_range[1] + .5, 1), np.log10(list), color=color)

        # Adding axes labels
        axes_counts[0][0].set_xlabel('Kp_value')
        axes_counts[0][0].set_ylabel('$log_{10}(counts)$')
        axes_counts[0][0].set_title('$log_{10}(counts)$')

    # Set the range of x values
    axes_counts[0][0].set_xlim(Kp_range[0], Kp_range[1])

    plt.show()


def counts_earth_plot(data, mode='cartesian', log=True):
    # This creates a plot, showing the number of data points measured in each bin around the Earth
    # This function only works with Kp bin sizes of 3. Like the others, could generalize at some point
    # I'm not sure polar actually works with this

    # Create some variables used for making the plot
    min_Lvalue = data['L'][0, 0].values
    max_Lvalue = data['L'][-1, 0].values
    nL = int(max_Lvalue - min_Lvalue + 1)
    nMLT = 24
    nbins = int(3)
    step = 9 / int(nbins)

    # Create the figures for the plots
    fig_counts, axes_counts = plt.subplots(nrows=1, ncols=nbins, squeeze=False, subplot_kw=dict(projection='polar'))
    plt.suptitle('Data points versus Location Around Earth')
    fig_counts.tight_layout()

    # This number is for helping to create the name of the variable in the thingy. Eg for step size 3, it goes from 0 to 3.0 to 6.0 to 9.0
    counter = 0

    # This is for creating the axes values for the plot
    phi = (2 * np.pi * data['MLT'].values / 24).reshape(nL, nMLT)
    r = data['L'].values.reshape(nL, nMLT)

    # List of axes labels
    counts_titles = ['Kp: [0,3)', 'Kp: [3,6)', 'Kp: [6,9]']

    if mode == 'polar':
        # Does polar even work? I don't think there's supposed to be 3 plots here
        for plot_number in range(nbins):
            # Designate which plot to plot on
            ax2 = axes_counts[0, plot_number]

            # Create title
            ax2.set_title(counts_titles[plot_number])

            # Get the count data for one Kp range
            count_data = data['E_GSE_polar_Kp_' + str(counter) + '_to_' + str(counter + step) + '_count']

            # Plot the values
            if log == False:
                im = ax2.pcolormesh(phi, r, count_data, cmap='OrRd', shading='auto',
                                    vmax=5)  # vmax value may need to be changed
            else:
                im = ax2.pcolormesh(phi, r, np.log10(count_data), cmap='OrRd', shading='auto',
                                    vmax=5)  # vmax value may need to be changed

            # Increase the counter so that on the next loop, the next Kp bin will be plotted
            counter += step
    else:
        for plot_number in range(nbins):
            # Designate which plot to plot on
            ax2 = axes_counts[0, plot_number]

            # Create title
            ax2.set_title(counts_titles[plot_number])

            # Get the count data for one Kp range
            count_data = data['E_GSE_Kp_' + str(counter) + '_to_' + str(counter + step) + '_count']

            # Plot the values
            if log == False:
                im = ax2.pcolormesh(phi, r, count_data, cmap='OrRd', shading='auto',
                                    vmax=5)  # vmax value may need to be changed
            else:
                im = ax2.pcolormesh(phi, r, np.log10(count_data), cmap='OrRd', shading='auto',
                                    vmax=5)  # vmax value may need to be changed

            # Increase the counter so that on the next loop, the next Kp bin will be plotted
            counter += step

    # Create the colorbar
    colorbar = fig_counts.colorbar(im, ax=axes_counts.ravel().tolist())

    # Label the colorbar
    if log == False:
        colorbar.set_label('Counts')
    else:
        colorbar.set_label('$log_{10}(counts)$')

    plt.show()


def efield_vs_kp_plot(data, mode='cartesian'):
    # Kp bin size must be 3

    # polar doesn't work. at least i dont think

    # Create some variables used for making the plot
    min_Lvalue = data['L'][0, 0].values
    max_Lvalue = data['L'][-1, 0].values
    nL = int(max_Lvalue - min_Lvalue + 1)
    nMLT = 24
    nbins = int(3)

    # Its 3
    step = 9 / int(nbins)

    # Helps with getting names of variables doing it this way
    counter = 0

    # This is the amount of graphs that this function will create. It depends on the number of coordinates (1 plot per coordinate). Polar has 2 (r, \theta), cartesian has 3 (x, y, z)
    if mode == 'cartesian':
        rounds = 3
    elif mode == 'polar':
        rounds = 2
    else:
        raise Exception('the mode should either be cartesian or polar. Default is cartesian')

    # This is for creating the axes values for the plot
    phi = (2 * np.pi * data['MLT'].values / 24).reshape(nL, nMLT)
    r = data['L'].values.reshape(nL, nMLT)

    # This plot is for the binned Kp values
    fig, axes = plt.subplots(nrows=nbins, ncols=rounds, squeeze=False, subplot_kw=dict(projection='polar'))
    fig.tight_layout()

    # This plot is for the full Kp values. May merge into one plot at some point
    fig_again, axes_again = plt.subplots(nrows=3, ncols=1, squeeze=False, subplot_kw=dict(projection='polar'))
    fig_again.tight_layout()

    # Iterate through all rows (number of bins aka 3), and then all columns (aka coordinates)
    for row in range(nbins):
        for col in range(rounds):
            ax = axes[row, col]
            if mode == 'polar':
                # List of coordinates
                list = ['r', 'phi']

                # the name of the variable to be plotted
                string = 'E_GSE_polar_Kp_' + str(counter) + '_to_' + str(counter + step) + '_mean'

                # plot the data
                im = ax.pcolormesh(phi, r, data[string].loc[:, :, list[col]], cmap='seismic', vmin=-2, vmax=2)

                # For making the title (may not work right for polar)
                new_string = string[9:-5].split('_')

                # set the title. only difference between row 2 and other is the last bracket instead of parenthesis
                if row == 2:
                    ax.set_title('E' + list[col] + ' Kp: [' + new_string[2] + ',' + new_string[4] + ']')
                else:
                    ax.set_title('E' + list[col] + ' Kp: [' + new_string[2] + ',' + new_string[4] + ')')

                # This is for the plot of electric field for all Kp values
                if row == 0:
                    axes_again[col][0].pcolormesh(phi, r, data['E_GSE_polar_mean'].loc[:, :, list[col]], cmap='seismic',
                                                  shading='auto', vmin=-2,
                                                  vmax=2)
                    axes_again[col][0].set_title('E' + list[col] + ' For All Kp')
            else:
                #  List of coordinates
                list = ['x', 'y', 'z']

                # the name of the variable to be plotted
                string = 'E_GSE_Kp_' + str(counter) + '_to_' + str(counter + step) + '_mean'

                # plot the data
                im = ax.pcolormesh(phi, r, data[string].loc[:, :, list[col]], cmap='seismic', shading='auto', vmin=-2,
                                   vmax=2)  # These values may need to be changed

                # For making a the title
                new_string = string[9:-5].split('_')
                if row == 2:
                    ax.set_title('E' + list[col] + ' Kp: [' + new_string[0] + ',' + new_string[2] + ']')
                else:
                    ax.set_title('E' + list[col] + ' Kp: [' + new_string[0] + ',' + new_string[2] + ')')

                # This is for the plot of electric field for all Kp values
                if row == 0:
                    axes_again[col][0].pcolormesh(phi, r, data['E_GSE_mean'].loc[:, :, list[col]], cmap='seismic',
                                                  shading='auto', vmin=-2,
                                                  vmax=2)
                    axes_again[col][0].set_title('E' + list[col] + ' For All Kp')

        # increase the count so next range of Kp can be plotted
        counter += step

    # create a colorbar. Note that doing this on both plots doesn't work as you would hope as the colorbars are different. Not sure how to get the same colorbar onto the second plot
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()