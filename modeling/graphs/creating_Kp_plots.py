import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


def plot_Kp(data, data2, nbins, nbins2, mode):
    # This whole function is written weird, its half generalized and half not.
    # It only works for bin values of 3 in data1 and 9 in data2, but that is all that we need for now, so I'm gonna leave it

    # This program needs a lot of gloss. It looks horrendous and needs to be rewritten so that it is actually readable.

    # Setting up constant values based on values in the datasets
    nbins = int(nbins)
    nbins2 = int(nbins2)
    min_Lvalue = data['L'][0, 0].values
    max_Lvalue = data['L'][-1, 0].values
    nL = int(max_Lvalue - min_Lvalue + 1)
    nMLT = 24

    if mode == 'cartesian':
        rounds=3
    else:
        rounds=2

    # Setting up plots for the electric field binned by Kp
    fig, axes = plt.subplots(nrows=nbins, ncols=rounds, squeeze=False, subplot_kw=dict(projection='polar'))
    fig.tight_layout()

    fig_again, axes_again = plt.subplots(nrows=3, ncols=1, squeeze=False, subplot_kw=dict(projection='polar'))
    fig_again.tight_layout()

    # Instantiating counter value and setting up the size of each bin
    counter = 0
    step = 9 / int(nbins)
    step2 = 9 / int(nbins2)

    phi = (2 * np.pi * data['MLT'].values / 24).reshape(nL, nMLT)
    r = data['L'].values.reshape(nL, nMLT)


    # Plotting electric field vs Kp
    for row in range(nbins):
        for col in range(rounds):
            ax = axes[row, col]
            if mode == 'polar':  # Doesnt work for polar, since it goes 3 times but only 2 coords
                list = ['r', 'phi']
                string = 'E_GSE_polar_Kp_' + str(counter) + '_to_' + str(counter + step) + '_mean'
                im = ax.pcolormesh(phi, r, data[string].loc[:, :, list[col]], cmap='seismic', vmin=-2, vmax=2)
                print(string)
                new_string = string[9:-5].split('_')
                if row == 2:
                    ax.set_title('E' + list[col] + ' Kp: [' + new_string[2] + ',' + new_string[4] + ']')
                else:
                    ax.set_title('E' + list[col] + ' Kp: [' + new_string[2] + ',' + new_string[4] + ')')
            else:
                list = ['x', 'y', 'z']
                string = 'E_GSE_Kp_' + str(counter) + '_to_' + str(counter + step) + '_mean'
                im = ax.pcolormesh(phi, r, data[string].loc[:, :, list[col]], cmap='seismic', shading='auto', vmin=-2,
                                   vmax=2)  # These values may need to be changed
                new_string = string[9:-5].split('_')
                if row == 2:
                    ax.set_title('E' + list[col] + ' Kp: [' + new_string[0] + ',' + new_string[2] + ']')
                else:
                    ax.set_title('E' + list[col] + ' Kp: [' + new_string[0] + ',' + new_string[2] + ')')

                if row==0:
                    axes_again[col][0].pcolormesh(phi, r, data['E_GSE_mean'].loc[:, :, list[col]], cmap='seismic', shading='auto', vmin=-2,
                                  vmax=2)
                    axes_again[col][0].set_title('E'+list[col]+' For All Kp')
        counter += step
    fig.colorbar(im, ax=axes.ravel().tolist())

    fig.savefig('testing.pdf', format='pdf')

    # For the counts
    fig_counts, axes_counts = plt.subplots(nrows=1, ncols=nbins, squeeze=False, subplot_kw=dict(projection='polar'))
    plt.suptitle('Data points versus Location Around Earth')
    fig_counts.tight_layout()

    counter_counts = 0

    counts_titles = ['Kp: [0,3)', 'Kp: [3,6)', 'Kp: [6,9]']
    if mode=='polar':
        for plot_number in range(nbins): #LOG10 COUNTS.
            ax2 = axes_counts[0, plot_number]
            ax2.set_title(counts_titles[plot_number])
            count_data = data['E_GSE_polar_Kp_' + str(counter_counts) + '_to_' + str(counter_counts + step) + '_count']
            im = ax2.pcolormesh(phi, r, np.log10(count_data), cmap='OrRd', shading='auto', vmax=5)  # This values may need to be changed
            counter_counts += step
    else:
        for plot_number in range(nbins): #LOG10 COUNTS.
            ax2 = axes_counts[0, plot_number]
            ax2.set_title(counts_titles[plot_number])
            count_data = data['E_GSE_Kp_' + str(counter_counts) + '_to_' + str(counter_counts + step) + '_count']
            im = ax2.pcolormesh(phi, r, np.log10(count_data), cmap='OrRd', shading='auto', vmax=5)  # This values may need to be changed
            counter_counts += step
    fig_counts.colorbar(im, ax=axes_counts.ravel().tolist())

    # Now the line plot
    fig2, axes2 = plt.subplots(nrows=1, ncols=rounds, squeeze=False)
    fig2.tight_layout()
    # plt.suptitle('Electric Field vs Distance from Earth')
    # print(data2['E_GSE_mean'])
    # string = 'E_GSE_Kp_1.0_to_2.0_mean'

    colors = ['purple', 'mediumpurple', 'cornflowerblue', 'lightseagreen', 'forestgreen', 'yellow', 'orange', 'red']
    labels = ['[0,1)', '[1,2)', '[2,3)', '[3,4)', '[4,5)', '[5,6)', '[6,7)', '[7,9]']
    axes = [['L', 'Ex (GSE)'], ['L', 'Ey (GSE)'], ['L', 'Ez (GSE)']]

    # The line plot for just Efield vs L
    # THIS DEFINITELY DOESNT WORK FOR OTHER BIN AMOUNTS. Kp_value does not correspond to counter & stuff. MAYBE OR MAYBE NOT FIX
    for component in range(rounds):
        counter2 = 0
        ax2 = axes2[0, component]
        ax2.set_xlim([min_Lvalue, max_Lvalue])
        ax2.set_xlabel(axes[component][0])
        ax2.set_ylabel(axes[component][1])
        for Kp_value in range(6):
            list = []
            if mode=='polar':
                string = 'E_GSE_polar_Kp_' + str(counter2) + '_to_' + str(counter2 + step2)
            else:
                string = 'E_GSE_Kp_' + str(counter2) + '_to_' + str(counter2 + step2)
            if Kp_value == 7:
                if mode=='polar':
                    string2 = 'E_GSE_polar_Kp_' + str(counter2 + step2) + '_to_' + str(counter2 + 2 * step2)
                else:
                    string2 = 'E_GSE_Kp_' + str(counter2 + step2) + '_to_' + str(counter2 + 2 * step2)
                for counter3 in range(nL):
                    list.append(weighted_average(
                        data2[string + '_mean'][counter3, :, component].values + data2[string2 + '_mean'][counter3, :, component].values,
                        data2[string + '_count'][counter3, :].values + data2[string2 + '_count'][counter3, :].values))
                    ax2.plot(data['iL'].values[counter3] + 4.5, weighted_average(
                        data2[string + '_mean'][counter3, :, component].values + data2[string2 + '_mean'][counter3, :, component].values,
                        data2[string + '_count'][counter3, :].values + data2[string2 + '_count'][counter3, :].values),
                             color=colors[Kp_value], marker="o", markersize=3)
                ax2.plot(data['iL'].values + 4.5, list, color=colors[Kp_value], label=labels[Kp_value])
            else:
                for counter3 in range(nL):
                    list.append(weighted_average(data2[string+'_mean'][counter3, :, component].values, data2[string+'_count'][counter3, :].values))
                    ax2.plot(data['iL'].values[counter3] + 4.5,
                             weighted_average(data2[string+'_mean'][counter3, :, component].values, data2[string+'_count'][counter3, :].values),
                             color=colors[Kp_value], marker="o", markersize=3)
                ax2.plot(data['iL'].values + 4.5, list, color=colors[Kp_value], label=labels[Kp_value])
            counter2 += step2
        if component == 0:
            fig2.legend(title="Kp Index")

    # plt.show()

    # This is fucked. I think its cause something isn't getting reset where it should be.
    # However I don't think that this is something I actually need to fix, since I will be using the other counts plot
    # fig2_counts, axes2_counts = plt.subplots(nrows=3, ncols=3, squeeze=False)
    # plt.suptitle('Number of Data Points in Each L Range')
    # counter2_counts = 0
    # for Kp_value_counts in range(8):
    #     list = []
    #     row=int(Kp_value_counts / 3)
    #     column = Kp_value_counts % 3
    #     ax2_counts = axes2_counts[row][column]
    #     string = 'E_GSE_Kp_' + str(counter2_counts) + '_to_' + str(counter2_counts + step2) + '_count'
    #     if Kp_value_counts == 7:
    #         string2 = 'E_GSE_Kp_' + str(counter2_counts + step2) + '_to_' + str(counter2_counts + 2 * step2) + '_count'
    #         for counter3_counts in range(nL):
    #             list.append(((data2[string][counter3_counts]+data2[string2][counter3_counts]).sum()))
    #         ax2_counts.bar(data['iL'].values + 4.5, list)
    #         ax2_counts.set_title(labels[Kp_value_counts])
    #     else:
    #         for counter3_counts in range(nL):
    #             list.append(((data2[string][counter3_counts]).sum()))
    #         ax2_counts.bar(data['iL'].values + 4.5, list)
    #         ax2_counts.set_title(labels[Kp_value_counts])

    # This is a single bar graph, with counts on the y axis and Kp value on the x axis
    # Specifically with data2 being 9 bins
    fig3_counts, axes3_counts = plt.subplots(nrows=1, ncols=1, squeeze=False)
    list = []
    for Kp_values in range (8):
        if Kp_values==0:
            string = 'E_GSE_Kp_' + str(Kp_values) + '_to_' + str(Kp_values+1) + '.0_count'
        else:
            string = 'E_GSE_Kp_' + str(Kp_values) + '.0_to_' + str(Kp_values + 1) + '.0_count'
        list.append(data2[string].values.sum())
    axes3_counts[0][0].set_xlabel('Kp_value')
    axes3_counts[0][0].set_ylabel('$log_{10}(counts)$')
    axes3_counts[0][0].set_title('$log_{10}(counts)$')
    axes3_counts[0][0].bar(np.arange(0.5, 8.5, 1), np.log10(list), color='blue')
    axes3_counts[0][0].set_xlim(0,9)
    plt.show()

def weighted_average(data, data_counts):
    total = sum(data * data_counts)
    total_counts = sum(data_counts)
    # Idk what else to do when I have 0 counts. If this isn't done then we get nan and no line printed
    if total_counts == 0:
        return 0
    else:
        return total/total_counts


def main():
    data = xr.open_dataset('6_years_Kp3.nc')
    data2 = xr.open_dataset('6_years_Kp9.nc')
    nbins = 3.0
    nbins2 = 9.0
    mode = 'cartesian'
    plot_Kp(data, data2, nbins, nbins2, mode)


if __name__ == '__main__':
    main()
