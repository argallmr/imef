import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import imef.data.data_manipulation as dm
import visualizations.plot_nc_data as xrplot
import visualizations.visualizations as vis


def create_histogram(data, index='Kp', bins=np.array([0, 1, 2, 3, 4, 5, 6, 7, 9]), checkmarks=np.array([.25, .5, .75, 1])):
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
    # matplotlib is limited here. maybe move to datashader or holoviews so that the binning can be done better (.5 just duplicates the values. Maybe numpy?)
    ax2 = axes[0][1]
    bins2 = np.arange(bins[0], bins[-1])
    returned = ax2.hist(data[index].values, bins2, histtype='step', cumulative=True, orientation='horizontal')
    counts = returned[0]
    x_ticks = returned[1]
    ax2.set_xlabel('Cumulative Number of Data Points')
    ax2.set_ylabel(index + " (nT)")

    total_counts = counts[-1]

    checkmark_counter = 0
    counts_counter = 0
    new_bins=np.array([-160])
    while checkmark_counter < len(checkmarks):
        count_marker = checkmarks[checkmark_counter] * total_counts
        if counts[counts_counter] >= count_marker:
            if int(checkmarks[checkmark_counter]) != 1:
                ax2.hlines(y=x_ticks[counts_counter], xmin=0, xmax=counts[counts_counter])
            ax2.vlines(x=counts[counts_counter], ymin=bins[0], ymax=x_ticks[counts_counter])
            new_bins = np.append(new_bins, x_ticks[counts_counter])
            checkmark_counter += 1
        counts_counter += 1


    # fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
    # this line doesn't seem to be working as intended
    # plt.xticks(bins)
    ax = axes[0][0]
    ax.hist(data[index].values, bins=bins)
    ax.set_ylabel('Number of Data Points')



    plt.show()


def plot_potential(V_data, L, MLT):
    # Note that it is expected that the electric field data is in polar coordinates. Otherwise the potential values are incorrect

    # find L and MLT range used in the data given
    min_Lvalue = L[0][0]
    max_Lvalue = L[-1][0]
    nL = int(max_Lvalue - min_Lvalue + 1)

    min_MLTvalue = 0.5
    max_MLTvalue = 23.5
    nMLT = 24

    # Create a coordinate grid
    new_values = MLT.values-1
    phi = (2 * np.pi * new_values / 24).reshape(nL, nMLT)
    r = L.values.reshape(nL, nMLT)-1

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
    # Plot the data. Note that new_V_data is multiplied by -1, since the L/MLT coordinate system has positive x and positive y in the opposite direction of normal cartesian coordinates
    im = ax1.contourf(new_phi, new_r, new_V_data*-1, cmap='coolwarm', vmin=-5, vmax=5)
    fig.colorbar(im, ax=ax1)
    # Draw the earth
    xrplot.draw_earth(ax1)

    plt.show()


def plot_efield(imef_data, plotted_variable, mode='cartesian', count=True, log_counts=False):

    # find L and MLT range used in the data given
    min_Lvalue = imef_data['r'][0].values
    max_Lvalue = imef_data['r'][-1].values
    nL = int(max_Lvalue - min_Lvalue + 1)

    # min_MLTvalue = imef_data['MLT'][0, 0].values
    # max_MLTvalue = imef_data['MLT'][0, -1].values
    nMLT = 24

    Lgrid, MLTgrid = xr.broadcast(imef_data['r'], imef_data['theta']*12/np.pi)

    # Create a coordinate grid
    phi = (2 * np.pi * MLTgrid.values / 24).reshape(nL, nMLT)
    r = Lgrid.values.reshape(nL, nMLT)
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
    if count==True:
        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, subplot_kw=dict(projection='polar'))
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, subplot_kw=dict(projection='polar'))
    fig.tight_layout()

    # Plot the electric field
    # Scale makes the arrows smaller/larger. Bigger number = smaller arrows.
    # May need to be changed when more data points are present
    ax1 = axes[0, 0]

    # Note that Ex and Ey are multiplied by -1, since the L/MLT coordinate system has positive x and positive y in the opposite direction as is standard
    ax1.quiver(phi, r, -1 * Ex, -1 * Ey, scale=5)
    ax1.set_xlabel("Electric Field")
    ax1.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    ax1.set_theta_direction(1)

    # Draw the earth
    xrplot.draw_earth(ax1)

    # Plot the number of data points in each bin
    if count==True:
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
        xrplot.draw_earth(ax2)

    plt.show()


def calculate_and_plot_potential(data):
    # Size of the bins
    data=data.rename({'comp':'cart'})
    dr = (data['r'][1]-data['r'][0]).values
    dtheta = (data['theta'][1]-data['theta'][0]).values*12/np.pi

    # Before I changed this, the r and theta were each 1 bigger than the r_bins and theta_bins. I'm pretty sure this is incorrect but should double check with Matt
    theta = np.arange(0.5, 23.5 + dtheta, dtheta) * 2 * np.pi / 24
    r = np.arange(0.5, 9.5 + dr, dr)

    r_bins = int((data['r'][-1] - data['r'][0] + 1).values)
    theta_bins = int(((data['theta'][-1] - data['theta'][0]) * 12 / np.pi + 1).values)
    number_of_bins = int(theta_bins * r_bins)

    # make a grid of values that will convert the electric field from cartesian to polar (cylindrical) coordinates
    ngrid = len(theta) * len(r)
    theta_grid, r_grid = np.meshgrid(theta, r)
    z_grid = np.zeros(ngrid)

    cyl_grid = xr.DataArray(np.stack([r_grid.flatten(), theta_grid.flatten(), z_grid], axis=1),
                            dims=('time', 'cyl'),
                            coords={'time': np.arange(ngrid),
                                    'cyl': ['r', 'phi', 'z']}
                            )
    cart_grid = dm.cyl2cart(cyl_grid)
    xcart2cyl = dm.xform_cart2cyl(cart_grid)

    # get the convective electric field
    E_convection = data['E_con_mean']

    # plot electric field
    # plot_efield(data, 'E_con_mean', count=False)

    # reshape the data so that it will work with the conversion function
    E_convection_reshaped = xr.DataArray(E_convection.values.reshape(number_of_bins, 3), dims=['time', 'cart'],
                                         coords={'time': np.arange(number_of_bins), 'cart': data['cart']})

    # convert to polar
    E_convection_polar = xcart2cyl.dot(E_convection_reshaped, dims='cart')

    # reshape and store the polar data in a new dataset
    imef_data = xr.Dataset()
    data_polar_split = xr.DataArray(E_convection_polar.values.reshape(r_bins, theta_bins, 3), dims=['r', 'theta', 'cart'],
                                    coords={'r': data['r'], 'theta': data['theta'] * (12 / np.pi),
                                            'cart': data['cart']})
    imef_data['E_con_polar'] = data_polar_split
    imef_data = imef_data.rename({'r': 'L', 'theta': 'MLT'})
    variable_name = 'E_con_polar'

    otherL, otherMLT = xr.broadcast(imef_data['L'], imef_data['MLT'])

    imef_data['E_conv_polar'] = imef_data['E_con_polar'].fillna(0)

    # calculate the electric potential and plot
    V = dm.calculate_potential(imef_data, variable_name)
    plot_potential(V, otherL, otherMLT)


def plot_global_efield_kp(ds, varname='E_EDI_corot'):
    '''
    Create a polar plot of the Electric field vectors binned in (r, theta) space.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Data to be plotted, including the r and theta bins.
    varname : str
        Name of the variable to be plotted

    Returns
    -------
    fig : matplotlib figure
        Figure object
    axes : matplotlib subplots
        Axes in which the counts and electric field are plotted
    '''

    fig, axes = plt.subplots(nrows=3, ncols=3, squeeze=False, figsize=(7, 7),
                             subplot_kw=dict(projection='polar'))
    plt.subplots_adjust(wspace=0.6, hspace=0.65, bottom=0.1)

    for idx, E in enumerate(ds[varname + '_mean']):
        irow = idx // 3
        icol = idx % 3
        ax = axes[irow, icol]

        ax_title = 'kp=[{0},{1}-]'.format(np.ceil(ds['Kp'][idx].data),
                                          np.ceil(ds['Kp'][idx].data) + 1)

        plot_global_efield_one(E, ds[varname + '_counts'][idx, ...],
                               axes=ax, plot_counts=False)

        ax.set_title(ax_title)

        if ~ax.is_last_row():
            ax.set_xlabel('')
        #            ax.set_xticklabels([])

        if ~ax.is_first_col():
            ax.set_ylabel('')

    # Last plot
    ax = axes[-1, -1]
    ax.set_title('Kp=[0,9-]')
    ax.set_xlabel('')

    # Average over the remaining theta bins
    data = xr.DataArray(np.ma.average(ds[varname + '_mean'].to_masked_array(), axis=0,
                                      weights=ds[varname + '_counts']),
                        dims=('r', 'theta', 'comp'),
                        coords={'r': ds['r'],
                                'theta': ds['theta'],
                                'comp': ds['comp']})

    # Sum all counts over the theta dimension
    cts = np.sum(ds[varname + '_counts'], axis=0)
    cts = cts.where(cts != 0)

    plot_global_efield_one(data, cts,
                           axes=ax, plot_counts=False)

    return fig, axes

def plot_global_efield_one(E, counts, axes=None, plot_counts=False):
    '''
    Create a polar plot of the Electric field vectors binned in (r, theta) space.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Data to be plotted, including the r and theta bins.
    varname : str
        Name of the variable to be plotted

    Returns
    -------
    fig : matplotlib figure
        Figure object
    axes : matplotlib subplots
        Axes in which the counts and electric field are plotted
    '''
    # Create the axes
    if axes is None:
        # Number of columns
        #   - Irrelevant if axes is given
        ncols = 1
        if plot_counts:
            ncols = 2

        # Create the figure
        fig, axes = plt.subplots(nrows=1, ncols=ncols, squeeze=False, figsize=(6.5, 4),
                                 subplot_kw=dict(projection='polar'))
        plt.subplots_adjust(wspace=0.33, bottom=0.15, right=0.85)

        # Assign the axes
        ax = axes[0, 0]
        if plot_counts:
            ax_cts = axes[0, 1]

    # Axes were given
    else:

        # Assign the axes
        if plot_counts:
            ax = axes[0]
            ax_cts = axes[1]
        else:
            ax = axes

        # Get the figure
        fig = ax.figure

    # Global Electric Field
    ax.quiver(E['theta'], E['r'], E.values[..., 0], E.values[..., 1])
    # ax.quiver(E['theta'], E['r'], E[..., 0], E[..., 1])
    ax.set_xlabel("Electric Field")
    ax.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    ax.set_theta_direction(1)

    # Draw the earth
    draw_earth_pol(ax)

    # Counts
    if plot_counts:
        plot_global_counts_one(counts[..., 0], axes=ax_cts)

    return fig, axes

def draw_earth_pol(ax):
    '''
    A handy function for drawing the Earth in a set of Polar Axes
    '''
    ax.fill_between(np.linspace(-np.pi / 2, np.pi / 2, 30), 0, np.ones(30), color='k')
    ax.plot(np.linspace(np.pi / 2, 3 * np.pi / 2, 30), np.ones(30), color='k')

def plot_global_counts_one(counts, axes=None):
    if axes is None:
        fix, axes = plt.subplots(nrows=1, ncols=1, squeeze=False,
                                 subplot_kw=dict(projection='polar'))
        ax = axes[0, 0]
    else:
        ax = axes
        fig = ax.figure

    log_counts = np.ma.log(counts)

    im = ax.pcolormesh(counts['theta'], counts['r'], log_counts,
                       cmap='YlOrRd', shading='auto')
    ax.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    ax.set_xlabel("log$_{10}$(Counts)")

    draw_earth_pol(ax)

    vis.add_colorbar(ax, im, wpad=1.3)

    return fig, axes


def main():
    data = xr.open_dataset('mms1_imef_srvy_l2_5sec_20150901000000_20220701000000.nc')
    data_binned_nk = xr.open_dataset('mms1_imef_srvy_l2_5sec_20150901000000_20220701000000_binned_r_theta.nc')
    data_binned = xr.open_dataset('mms1_imef_srvy_l2_5sec_20150901000000_20220701000000_binned_r_theta_Kp.nc')
    # data_binned_nt = xr.open_dataset('mms1_imef_srvy_l2_5sec_20150901000000_20220701000000_binned_r_Kp.nc')

    # data_binned = xr.open_dataset('mms1_imef_srvy_l2_5sec_20150901000000_20220701000000_binned_r_theta_Dst.nc')

    # data_binned = xr.open_dataset('mms1_imef_srvy_l2_5sec_20150901000000_20220701000000_binned_r_theta_AL.nc')

    # data_binned = xr.open_dataset('mms1_imef_srvy_l2_5sec_20150901000000_20220701000000_binned_r_theta_IEF.nc')

    # There is none in 40-60 for Dst, but is for Sym-H
    # create_histogram(data, index='Sym-H', bins=np.arange(-140, 60, 2))
    #
    # calculate_and_plot_potential(data_binned_nk)
    #
    # fig, axes = vis.plot_global_efield_one(data_binned_nk, None)
    # plt.show()
    #
    # fig, axes = vis.plot_global_counts_index(data_binned, varname='E_con', index='Kp')
    # plt.show()

    # print(data_binned['AL'])

    # fig, axes = vis.plot_efield_r_index(data_binned, 'E_con', index='Kp')

    plot_global_efield_kp(data_binned, varname='E_con')
    plt.show()


if __name__ == '__main__':
    main()