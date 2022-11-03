import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Circle
from matplotlib.patches import Wedge


def draw_earth(ax, rad=1, angle=0, ecolor="w", polar=False, style=False, **kwargs):
    """
    A handy function for drawing stylistic dayside/nightside Earth on either cartesian or polar axis.

    Args:
        ax (axis): current axis
        rad (float): radius of projected Earth. If units [RE], use 1. Defaults to 1.
        angle (int, optional): rotation in [deg]. Defaults to 0.
        ecolor (str, optional): color of earth. Defaults to 'w' (white).
        polar (bool, optional): choice of prjection on cartesian (False) or polar (True) axis. Defaults to False.
        style (bool, optional): turns on monotone hatch styling. Defaults to False.
    """

    theta1 = 90 + angle
    theta2 = 270 + angle

    hstyle = "/////"

    # plot on cartesian axis
    if polar == False:

        loc = (0, 0)

        # create semicircles
        dayside = Wedge(
            loc, rad, theta1, theta2, facecolor=ecolor, edgecolor="k", **kwargs
        )
        nightside = Wedge(
            loc, rad, theta2, theta1, facecolor="k", edgecolor="k", **kwargs
        )

        # plot
        ax.add_artist(dayside)
        ax.add_artist(nightside)

    # plot on polar axis
    else:

        # fill color
        dayside = ax.fill_between(
            np.linspace(np.radians(theta1), np.radians(theta2), 30),
            0,
            np.ones(30) * rad,
            color=ecolor,
            **kwargs
        )
        nightside = ax.fill_between(
            np.linspace(-np.radians(theta1), np.radians(theta1), 30),
            0,
            np.ones(30) * rad,
            color="k",
            **kwargs
        )

        # outline
        ax.plot(
            np.linspace(np.radians(theta1), np.radians(theta2), 30),
            np.ones(30) * rad,
            color="k",
            **kwargs
        )
        ax.plot(
            np.linspace(-np.radians(theta1), np.radians(theta1), 30),
            np.ones(30) * rad,
            color="k",
            **kwargs
        )

    # hatch style shading
    if style == True:
        nightside.set_fc(ecolor)
        nightside.set_hatch(hstyle)
