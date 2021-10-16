import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


def plot_Kp(data, data2, nbins, nbins2, mode):
    # This whole function is written weird, its half generalized and half not.
    # It only works for bin values of 3 in data1 and 9 in data2, but that is all that we need for now, so I'm gonna leave it

    # Setting up constant values based on values in the datasets
    nbins = int(nbins)
    nbins2 = int(nbins2)
    min_Lvalue = data['L'][0, 0].values
    max_Lvalue = data['L'][-1, 0].values
    nL = int(max_Lvalue - min_Lvalue + 1)
    nMLT = 24

    # Setting up plots for the electric field binned by Kp
    fig, axes = plt.subplots(nrows=nbins, ncols=3, squeeze=False, subplot_kw=dict(projection='polar'))
    fig.tight_layout()

    # Instantiating counter value and setting up the size of each bin
    counter = 0
    step = 9 / int(nbins)
    step2 = 9 / int(nbins2)

    phi = (2 * np.pi * data['MLT'].values / 24).reshape(nL, nMLT)
    r = data['L'].values.reshape(nL, nMLT)

    # Plotting electric field vs Kp
    for row in range(nbins):
        for col in range(3):
            ax = axes[col, row]
            if mode == 'polar':  # Doesnt work for polar, since it goes 3 times but only 2 coords
                list = ['r', 'phi']
                string = 'E_GSE_polar_Kp_' + str(counter) + '_to_' + str(counter + step) + '_mean'
                im = ax.pcolormesh(phi, r, data[string].loc[:, :, list[col]], cmap='seismic')
                new_string = string[9:-5].split('_')
                if row == 2:
                    ax.set_title('E' + list[col] + ' Kp: [' + new_string[0] + ',' + new_string[2] + ']')
                else:
                    ax.set_title('E' + list[col] + ' Kp: [' + new_string[0] + ',' + new_string[2] + ')')
            else:
                list = ['x', 'y', 'z']
                string = 'E_GSE_Kp_' + str(counter) + '_to_' + str(counter + step) + '_mean'
                im = ax.pcolormesh(phi, r, data[string].loc[:, :, list[col]], cmap='seismic', shading='auto', vmin=-.5,
                                   vmax=.5)  # These values may need to be changed
                new_string = string[9:-5].split('_')
                if row == 2:
                    ax.set_title('E' + list[col] + ' Kp: [' + new_string[0] + ',' + new_string[2] + ']')
                else:
                    ax.set_title('E' + list[col] + ' Kp: [' + new_string[0] + ',' + new_string[2] + ')')
        counter += step
    fig.colorbar(im, ax=axes.ravel().tolist())

    # For the counts
    fig_counts, axes_counts = plt.subplots(nrows=1, ncols=nbins, squeeze=False, subplot_kw=dict(projection='polar'))
    fig_counts.tight_layout()

    counter_counts = 0

    for plot_number in range(nbins):
        ax2 = axes_counts[0, plot_number]
        count_data = data['E_GSE_Kp_' + str(counter_counts) + '_to_' + str(counter_counts + step) + '_count']
        im = ax2.pcolormesh(phi, r, count_data, cmap='OrRd', shading='auto',
                            vmax='1200')  # These values may need to be changed

        counter_counts += step
    fig_counts.colorbar(im, ax=axes_counts.ravel().tolist())

    # Now the line plot
    fig2, axes2 = plt.subplots(nrows=1, ncols=3, squeeze=False)
    plt.suptitle('Electric Field vs Distance from Earth')
    # print(data2['E_GSE_mean'])
    # string = 'E_GSE_Kp_1.0_to_2.0_mean'

    colors = ['purple', 'mediumpurple', 'cornflowerblue', 'lightseagreen', 'forestgreen', 'yellow', 'orange', 'red']
    labels = ['[0,1)', '[1,2)', '[2,3)', '[3,4)', '[4,5)', '[5,6)', '[6,7)', '[7,9]']
    axes = [['L', 'Ex (GSE)'], ['L', 'Ey (GSE)'], ['L', 'Ez (GSE)']]

    # THIS DEFINITELY DOESNT WORK FOR OTHER BIN AMOUNTS. Kp_value does not correspond to counter & stuff. MAYBE OR MAYBE NOT FIX
    for component in range(3):
        counter2 = 0
        ax2 = axes2[0, component]
        ax2.set_xlim([min_Lvalue, max_Lvalue])
        ax2.set_xlabel(axes[component][0])
        ax2.set_ylabel(axes[component][1])
        for Kp_value in range(8):
            list = []
            string = 'E_GSE_Kp_' + str(counter2) + '_to_' + str(counter2 + step2) + '_mean'
            if Kp_value == 7:
                string2 = 'E_GSE_Kp_' + str(counter2 + step2) + '_to_' + str(counter2 + 2 * step2) + '_mean'
                for counter3 in range(nL):
                    # print((data2[string][counter3, :, component].values+data2[string2][counter3, :, component].values).mean())
                    list.append((data2[string][counter3, :, component].values + data2[string2][counter3, :,
                                                                                component].values).mean())
                    ax2.plot(data['iL'].values[counter3] + 4.5, data2[string][counter3, :, component].values.mean(),
                             color=colors[Kp_value], marker="o", markersize=3)
                ax2.plot(data['iL'].values + 4.5, list, color=colors[Kp_value], label=labels[Kp_value])
            else:
                for counter3 in range(nL):
                    list.append(data2[string][counter3, :, component].values.mean())
                    ax2.plot(data['iL'].values[counter3] + 4.5, data2[string][counter3, :, component].values.mean(),
                             color=colors[Kp_value], marker="o", markersize=3)
                ax2.plot(data['iL'].values + 4.5, list, color=colors[Kp_value], label=labels[Kp_value])
            counter2 += step2
        if component == 0:
            fig2.legend(title="Kp Index")

    fig2_counts, axes2_counts = plt.subplots(nrows=3, ncols=3, squeeze=False)
    plt.suptitle('Number of Data Points in Each L Range')
    counter2_counts = 0
    for Kp_value_counts in range(8):
        list = []
        row=int(Kp_value_counts / 3)
        column = Kp_value_counts % 3
        ax2_counts = axes2_counts[row][column]
        string = 'E_GSE_Kp_' + str(counter2_counts) + '_to_' + str(counter2_counts + step2) + '_count'
        if Kp_value_counts == 7:
            string2 = 'E_GSE_Kp_' + str(counter2_counts + step2) + '_to_' + str(counter2_counts + 2 * step2) + '_count'
            # IMPLEMENT
        else:
            for counter3_counts in range(nL):
                list.append(((data2[string][counter3_counts]).sum()))
            ax2_counts.bar(data['iL'].values + 4.5, list)
            ax2_counts.set_title(labels[Kp_value_counts])

    plt.show()


def main():
    data = xr.open_dataset('6_years_Kp3.nc')
    data2 = xr.open_dataset('6_years_Kp9.nc')
    nbins = 3.0
    nbins2 = 9.0
    mode = 'cartesian'
    plot_Kp(data, data2, nbins, nbins2, mode)


if __name__ == '__main__':
    main()
