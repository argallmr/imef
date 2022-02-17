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


def plot_efield_cartesian(nL, nMLT, imef_data, plotted_variable, log=False):

    # Create a coordinate grid
    phi = (2 * np.pi * imef_data['MLT'].values / 24).reshape(nL, nMLT)
    r = imef_data['L'].values.reshape(nL, nMLT)
    Ex = imef_data[plotted_variable].loc[:, :, 'x'].values.reshape(nL, nMLT)
    Ey = imef_data[plotted_variable].loc[:, :, 'y'].values.reshape(nL, nMLT)

    # Plot the data
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, subplot_kw=dict(projection='polar'))

    # Plot the electric field
    # Scale makes the arrows smaller/larger. Bigger number = smaller arrows.
    # May need to be changed when more data points are present
    ax1 = axes[0, 0]
    ax1.quiver(phi, r, Ex, Ey, scale=14)
    ax1.set_xlabel("Electric Field")
    ax1.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])

    # Draw the earth
    draw_earth(ax1)

    # Plot the number of data points in each bin
    ax2 = axes[0, 1]
    ax2.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    ax2.set_xlabel("Count")
    if log==True:
        im = ax2.pcolormesh(phi, r, np.log10(imef_data['E_GSE_count'].data), cmap='YlOrRd', shading='auto')
    else:
        im = ax2.pcolormesh(phi, r, imef_data['E_GSE_count'].data, cmap='YlOrRd', shading='auto')
    fig.colorbar(im, ax=ax2)
    draw_earth(ax2)

    plt.show()


def plot_efield_polar(nL, nMLT, imef_data, plotted_variable, log=False):  # Update this to spherical if needed

    # Create a coordinate grid
    phi = (2 * np.pi * imef_data['MLT'].values / 24).reshape(nL, nMLT)
    r = imef_data['L'].values.reshape(nL, nMLT)
    Er = imef_data[plotted_variable].loc[:, :, 'r'].values.reshape(nL, nMLT)
    Ephi = imef_data[plotted_variable].loc[:, :, 'phi'].values.reshape(nL, nMLT)

    # Convert to cartesian coordinates
    # Scaling the vectors doesn't work correctly unless this is done.
    Ex = Er * np.cos(phi) - Ephi * np.sin(phi)
    Ey = Er * np.sin(phi) + Ephi * np.cos(phi)

    # Plot the data
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, subplot_kw=dict(projection='polar'))
    fig.tight_layout()

    # Plot the electric field
    # Scale makes the arrows smaller/larger. Bigger number = smaller arrows.
    # May need to be changed when more data points are present
    ax1 = axes[0, 0]
    ax1.quiver(phi, r, Ex, Ey, scale=14)
    ax1.set_xlabel("Electric Field")
    ax1.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    ax1.set_theta_direction(1)

    # Draw the earth
    draw_earth(ax1)

    # Plot the number of data points in each bin
    ax2 = axes[0, 1]
    ax2.set_thetagrids(np.linspace(0, 360, 9), labels=['0', '3', '6', '9', '12', '15', '18', '21', ' '])
    ax2.set_xlabel("Count")
    if log==True:
        im = ax2.pcolormesh(phi, r, np.log10(imef_data['E_GSE_polar_count'].data), cmap='YlOrRd', shading='auto')
    else:
        im = ax2.pcolormesh(phi, r, imef_data['E_GSE_polar_count'].data, cmap='YlOrRd', shading='auto')
    fig.colorbar(im, ax=ax2)
    draw_earth(ax2)

    plt.show()


def plot_potential(nL, nMLT, imef_data, V_data):

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
    im = ax1.contourf(new_phi, new_r, new_V_data, cmap='coolwarm', vmin=-25, vmax=25)
    # plt.clabel(im, inline=True, fontsize=8)
    # plt.imshow(new_V_data, extent=[-40, 12, 0, 10], cmap='RdGy', alpha=0.5)
    fig.colorbar(im, ax=ax1)
    # Draw the earth
    draw_earth(ax1)

    plt.show()