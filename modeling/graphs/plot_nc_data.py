import numpy as np
from matplotlib import pyplot as plt

# For debugging purposes
np.set_printoptions(threshold=np.inf)

def draw_earth(ax):
    '''
    A handy function for drawing the Earth in a set of Polar Axes
    '''
    ax.fill_between(np.linspace(-np.pi / 2, np.pi / 2, 30), 0, np.ones(30), color='k')
    ax.plot(np.linspace(np.pi / 2, 3 * np.pi / 2, 30), np.ones(30), color='k')


def plot_cartesian_data(nL, nMLT, imef_data):

    # Create a coordinate grid
    phi = (2 * np.pi * imef_data['MLT'].values / 24).reshape(nL, nMLT)
    r = imef_data['L'].values.reshape(nL, nMLT)
    Ex = imef_data['E_mean'].loc[:, :, 'x'].values.reshape(nL, nMLT)
    Ey = imef_data['E_mean'].loc[:, :, 'y'].values.reshape(nL, nMLT)

    # Plot the data
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, subplot_kw=dict(projection='polar'))

    # Plot the electric field
    # Scale makes the arrows smaller/larger. Bigger number = smaller arrows.
    # May need to be changed when more data points are present
    ax1 = axes[0, 0]
    ax1.quiver(phi, r, Ex, Ey, scale=10)
    ax1.set_xlabel("Electric Field")
    ax1.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])

    # Draw the earth
    draw_earth(ax1)

    # Plot the number of data points in each bin
    ax2 = axes[0, 1]
    ax2.set_xlabel("Count")
    im = ax2.pcolormesh(phi, r, imef_data['count'].data, cmap='YlOrRd', shading='auto')
    fig.colorbar(im, ax=ax2)

    plt.show()


def plot_polar_data(nL, nMLT, imef_data):  # Update this to spherical if needed

    # Create a coordinate grid
    phi = (2 * np.pi * imef_data['MLT'].values / 24).reshape(nL, nMLT)
    r = imef_data['L'].values.reshape(nL, nMLT)
    Er = imef_data['E_mean'].loc[:, :, 'r'].values.reshape(nL, nMLT)
    Ephi = imef_data['E_mean'].loc[:, :, 'phi'].values.reshape(nL, nMLT)

    # Convert to cartesian coordinates
    # Scaling the vectors doesn't work correctly unless this is done.
    Ex = Er * np.cos(phi) - Ephi * np.sin(phi)
    Ey = Er * np.sin(phi) + Ephi * np.cos(phi)

    # Plot the data
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, subplot_kw=dict(projection='polar'))

    # Plot the electric field
    # Scale makes the arrows smaller/larger. Bigger number = smaller arrows.
    # May need to be changed when more data points are present
    ax1 = axes[0, 0]
    ax1.quiver(phi, r, Ex, Ey, scale=10)
    ax1.set_xlabel("Electric Field")
    ax1.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    ax1.set_theta_direction(1)

    # Draw the earth
    draw_earth(ax1)

    # Plot the number of data points in each bin
    ax2 = axes[0, 1]
    ax2.set_xlabel("Count")
    im = ax2.pcolormesh(phi, r, imef_data['count'].data, cmap='YlOrRd', shading='auto')
    fig.colorbar(im, ax=ax2)

    plt.show()

# For debugging purposes
# def main():
#     import xarray as xr
#     L_range = (0, 25)
#     MLT_range = (0, 24)
#     dL = 1  # RE
#     dMLT = 1  # MLT
#     L = xr.DataArray(np.arange(L_range[0], L_range[1], dL), dims='L')
#     MLT = xr.DataArray(np.arange(MLT_range[0], MLT_range[1], dMLT), dims='MLT')
#     # Number of points in each coordinate
#     nL = len(L)
#     nMLT = len(MLT)
#     imef_data = xr.open_dataset('edi_data.nc')
#     plot_polar_data(nL, nMLT, imef_data)
#
# if __name__ == '__main__':
#     main()