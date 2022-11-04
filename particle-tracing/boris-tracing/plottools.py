import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Circle
from matplotlib.patches import Wedge
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


def draw_earth(ax, rad=1, angle=0, ecolor="w", polar=False, style=False, **kwargs):
    """
    A handy function for drawing stylistic dayside/nightside Earth on either cartesian or polar axis.
    Built upon M. Argall's draw_earth.

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


def draw_earth3D(ax, rad=1, ecolor="b", **kwargs):
    """
    Draw earth on 3D axis.

    Args:
        ax (axis): current axis
        rad (float, optional): radius of projected Earth. If units [RE], use 1. Defaults to 1.
        ecolor (str, optional): color of earth. Defaults to 'b' (blue).
    """
    ns = 100  # number samples
    stride = 1

    # phi; goes from [0:2*PI]
    phi = np.linspace(0.0, 2.0 * np.pi, ns)

    # theta; goes from [0:PI]
    tht = np.linspace(0.0, np.pi, ns)

    x = rad * np.outer(np.cos(phi), np.sin(tht))
    y = rad * np.outer(np.sin(phi), np.sin(tht))
    z = rad * np.outer(np.ones(np.size(phi)), np.cos(tht))

    ax.plot_surface(
        x, y, z, linewidth=0.0, cstride=stride, rstride=stride, color=ecolor, **kwargs
    )


def add_offscale_RE(ax, lmax, offset=0.1, **kwargs):
    """
    Draws offscale RE axis on polar plot.
    Based on stack overflow question 16605137.
    """
    # offset below plot
    # offset = 0.15

    # create new axis and set postion
    rect = ax.get_position()
    rect = (
        rect.xmin + rect.width / 2,
        rect.ymin - offset,
        rect.width / 2,
        rect.height / 2,
    )
    r_ax = ax.figure.add_axes(rect)

    # hide un-needed elements
    for loc in ["right", "left", "top"]:
        r_ax.spines[loc].set_visible(False)
        r_ax.tick_params(left=False, labelleft=False)
        r_ax.patch.set_visible(False)

    # set axis limits
    # r_ax.set_xlim([0,lmax])
    ticks = np.arange(0, lmax + 1, 2)
    plt.xticks(ticks)

    # initialize minor ticks
    r_ax.minorticks_on()
    r_ax.xaxis.set_minor_locator(MultipleLocator(1))

    # remove y-axis minor ticks
    r_ax.yaxis.set_tick_params(which="minor", bottom=False)

    # stylizing
    r_ax.spines["bottom"].set_linestyle("dotted")
    r_ax.tick_params(axis="x", length=8, which="major", direction="inout", **kwargs)
    r_ax.tick_params(axis="x", length=5, which="minor", direction="inout", **kwargs)
    r_ax.set_xlabel("[$R_E$]")
