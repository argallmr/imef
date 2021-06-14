import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Wedge

# Read the data file 6*24 = number of bins
nr = 6
nphi = 24
df = pd.read_csv('polar_test.csv')
print(df.head)

# Create a coordinate grid
phi = (2 * np.pi * df['MLT'] / 24).to_numpy().reshape(nr, nphi)
r = df['L'].to_numpy().reshape(nr, nphi)
Er = df['ER'].to_numpy().reshape(nr, nphi)
Ephi = df['EAZ'].to_numpy().reshape(nr, nphi)

# Convert to cartesian coordinates
Ex = Er*np.cos(phi) - Ephi*np.sin(phi)
Ey = Er*np.sin(phi) + Ephi*np.cos(phi)

# Convert to cartesian vectors
x = r * np.cos(phi)
y = r * np.sin(phi)

def draw_earth(ax):
    '''
    A handy function for drawing the Earth in a set of Polar Axes
    '''
    ax.fill_between(np.linspace(-np.pi/2, np.pi/2, 30), 0, np.ones(30), color='k')
    ax.plot(np.linspace(np.pi/2, 3*np.pi/2, 30), np.ones(30), color='k')


# Plot the data
fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, subplot_kw=dict(projection='polar'))

# Plot the electric field
ax = axes[0,0]

# Plot the electric field
ax.quiver(phi, r, Ex, Ey)
ax.set_xlabel("Electric Field")
ax.set_thetagrids(np.linspace(0, 360, 8), labels=['0', '3', '6', '9', '12', '15', '18', '21'])
ax.set_theta_direction(1)

# Create the Earth
draw_earth(ax)

plt.show()