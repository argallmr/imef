import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import imef.data.data_manipulation as dm
import visualizations.plot_nc_data as xrplot


def kp_histogram(data):
    # log10 is more difficult than it seems, since it bins the new log10 values, making the bins inaccurate
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
    ax=axes[0][0]
    bins = np.array([0,1,2,3,4,5,6,7,9])
    ax.hist(data['Kp'].values, bins=np.arange(0, 9))
    plt.xticks(np.arange(0, 10))
    axes[0][1].hist(data['Kp'].values, bins=bins)
    ax.set_xlabel('Kp Bin Sizes')
    ax.set_ylabel("# of Data Points in Each Kp Range")
    plt.xticks(np.arange(0, 10))

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
    E_convection = data['E_convective_mean']

    # plot electric field
    plot_efield(data, 'E_convective_mean', count=False)

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
    imef_data['E_convection_polar'] = data_polar_split
    imef_data = imef_data.rename({'r': 'L', 'theta': 'MLT'})
    variable_name = 'E_convection_polar'

    otherL, otherMLT = xr.broadcast(imef_data['L'], imef_data['MLT'])

    imef_data['E_convection_polar'] = imef_data['E_convection_polar'].fillna(0)

    # calculate the electric potential and plot
    V = dm.calculate_potential(imef_data, variable_name)
    plot_potential(V, otherL, otherMLT)


def main():
    data = xr.open_dataset('mms1_imef_srvy_l2_5sec_20150915000000_20210101000000.nc')
    data_binned = xr.open_dataset('mms1_imef_srvy_l2_5sec_20150915000000_20210101000000_binned_r_theta.nc')

    # kp_histogram(data)

    calculate_and_plot_potential(data_binned)

    # Redo calculate and plot potential with matt's plotting (if it exists), and compare. (mainly efield, kinda sketchy)




if __name__ == '__main__':
    main()