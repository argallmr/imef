import numpy as np
import math
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Circle
from matplotlib.patches import Wedge
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from boris import *

# colorscales for color consistency
seq_standard = "YlGnBu"
div_standard = "RdGy"
# pcolor = 'GnBu'

# UNH logo hexcode
nh_blue = "#233E8A"
white = ["#FFFFFF"]

# UNH-themed sequential colors
nh_blues = ["#152451", "#233E8A", "#3B62CE", "#7C97DE", "#BDCBEF", "#DFE5F7", "#EFF2FB"]
nh_grays = ["#333333", "#575757", "#858585", "#ADADAD", "#D6D6D6", "#EBEBEB", "#F5F5F5"]
nh_blues_r = nh_blues[::-1]
nh_grays_r = nh_grays[::-1]

# UNH-themed divergent colors
nh_cscale = nh_blues + white + nh_grays_r
nh_cscale_r = nh_cscale[::-1]

nh_cmap_div = LinearSegmentedColormap.from_list("cdiv", nh_cscale, N=500)
nh_cmap_div_r = LinearSegmentedColormap.from_list("cdiv_r", nh_cscale_r, N=500)
nh_cmap_seq = LinearSegmentedColormap.from_list("cseq", nh_blues, N=500)
nh_cmap_seq_r = LinearSegmentedColormap.from_list("cseq_r", nh_blues_r, N=500)


def save_figure(floc, fname):
    plt.savefig(floc + fname + ".pdf", bbox_inches="tight", dpi=400, transparent=True)


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


def add_offscale_RE(ax, lmax, offset=0.1, half=False, **kwargs):

    """
    Draws offscale RE axis on polar plot.
    Based on stack overflow question 16605137.
    """
    # offset below plot
    # offset = 0.15

    # create new axis and set postion
    rect = ax.get_position()

    if half == True:
        rect = (
            rect.xmin + rect.width / 2,
            rect.ymin - offset,
            rect.width / 2,
            rect.height / 2,
        )
    else:
        rect = (
            rect.xmin,
            rect.ymin - offset,
            rect.width,
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
    ticks = np.arange(0, lmax + 1, 2) if half == True else np.arange(-lmax, lmax + 1, 2)
    plt.xticks(ticks, [abs(i) for i in ticks])

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


def center_axis(ax):
    """
    Set axis at origin.
    """
    # move left y-axis and bottim x-axis to origin
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")

    # remove extra axis
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # select axis tick locations
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # minor ticks
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    # tick styling
    for axi in ["x", "y"]:
        ax.tick_params(axis=axi, length=6, which="major", direction="inout")
        ax.tick_params(axis=axi, length=4, which="minor", direction="inout")

    # endcaps on axis (source: Stack Overflow question 33737736)
    ax.plot(
        (1),
        (0),
        marker="|",
        ms=11,
        color="k",
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )
    ax.plot(
        (0),
        (1),
        marker="_",
        ms=11,
        color="k",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )
    ax.plot(
        (0),
        (0),
        marker="|",
        ms=11,
        color="k",
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )
    ax.plot(
        (0),
        (0),
        marker="_",
        ms=11,
        color="k",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )


def smallcoords(ax, dim, hx="x", vx="y", hc="k", vc="k"):

    # horizontal
    ax.plot([-dim, -dim + 2], [dim - 3, dim - 3], color=hc, lw=1)
    ax.annotate(
        hx,
        xy=(-dim, dim - 3),
        xycoords="data",
        xytext=(-dim + 3, dim - 3),
        textcoords="data",
        horizontalalignment="right",
        verticalalignment="center",
        color=hc,
    )

    # verical
    ax.plot([-dim, -dim], [dim - 3, dim - 1], color=vc, lw=1)
    ax.annotate(
        vx,
        xy=(-dim, -dim + 2),
        xycoords="data",
        xytext=(-dim, dim),
        textcoords="data",
        horizontalalignment="center",
        verticalalignment="top",
        color=vc,
    )


def tracingplot_2d(rdat, ax=None, axcenter=True, savefig=True, floc="../bin/"):

    if ax is None:
        fig, ax = plt.subplots(1, 3)
        fig.tight_layout(pad=0.5)

    fig.set_figheight(16)
    fig.set_figwidth(16)

    # radius of earth [m]
    RE = 6371000

    # scale coordinates in units [RE]
    rscale = rdat / RE

    ranges = []
    ranges.append(math.ceil(max(abs(rscale[:, 0]))))
    ranges.append(math.ceil(min(abs(rscale[:, 0]))))
    ranges.append(math.ceil(max(abs(rscale[:, 1]))))
    ranges.append(math.ceil(min(abs(rscale[:, 1]))))
    ranges.append(math.ceil(max(abs(rscale[:, 2]))))
    ranges.append(math.ceil(min(abs(rscale[:, 2]))))

    axrng = max(ranges)

    # y-x plot
    ax[0].plot(rscale[:, 1], rscale[:, 0], "k", linewidth=0.5)
    ax[0].set_xlabel("$y$ [$R_E$]")
    ax[0].set_ylabel("$x$ [$R_E$]")
    ax[0].add_patch(Circle((0, 0), 1.0, color="b", zorder=10))

    # fix axis
    ax[0].set_xlim([-axrng, axrng])
    ax[0].set_ylim([-axrng, axrng])
    ax[0].set_aspect("equal")

    # z-x plol
    ax[1].plot(rscale[:, 2], rscale[:, 0], "k", linewidth=0.5)
    ax[1].set_xlabel("$z$ [$R_E$]")
    ax[1].set_ylabel("$x$ [$R_E$]")
    ax[1].add_patch(Circle((0, 0), 1.0, color="b", zorder=10))

    # fix axis
    ax[1].set_xlim([-axrng, axrng])
    ax[1].set_ylim([-axrng, axrng])
    ax[1].set_aspect("equal")

    # z-y plot
    ax[2].plot(rscale[:, 2], rscale[:, 1], "k", linewidth=0.5)
    ax[2].set_xlabel("$z$ [$R_E$]")
    ax[2].set_ylabel("$y$ [$R_E$]")
    ax[2].add_patch(Circle((0, 0), 1.0, color="b", zorder=10))

    # fix axis
    ax[2].set_xlim([-axrng, axrng])
    ax[2].set_ylim([-axrng, axrng])
    ax[2].set_aspect("equal")

    draw_earth(ax[0], ecolor="w", zorder=10)
    draw_earth(ax[1], ecolor="w", zorder=10)
    draw_earth(ax[2], ecolor="w", zorder=10)

    # ax.plot(rscale[0][0],rscale[0][1],rscale[0][2],'r',marker='*', markersize=10)

    # plot starting points
    ax[0].plot(rscale[0][1], rscale[0][0], "r", marker="*", markersize=11, zorder=9)
    ax[1].plot(rscale[0][2], rscale[0][0], "r", marker="*", markersize=11, zorder=9)
    ax[2].plot(rscale[0][2], rscale[0][1], "r", marker="*", markersize=11, zorder=9)

    """
        ax[0].plot(
        rscale[0][1],
        rscale[0][0],
        "y",
        marker="*",
        markersize=13,
        markeredgecolor="k",
        markeredgewidth=0.25,
        zorder=9,
    )
    
    """

    datmax = 10
    ticks = np.arange(-datmax, datmax + 1, 2)

    ax[0].set_xticks(ticks, [abs(i) for i in ticks])
    ax[0].set_yticks(ticks, [abs(i) for i in ticks])
    ax[1].set_xticks(ticks, [abs(i) for i in ticks])
    ax[1].set_yticks(ticks, [abs(i) for i in ticks])
    ax[2].set_xticks(ticks, [abs(i) for i in ticks])
    ax[2].set_yticks(ticks, [abs(i) for i in ticks])

    if axcenter == True:
        # y-x
        center_axis(ax[0])
        ax[0].set_xlabel(" ")
        ax[0].set_ylabel(" ")
        smallcoords(ax[0], 10, hx="y", vx="x", hc="lime", vc="r")

        # z-x
        center_axis(ax[1])
        ax[1].set_xlabel(" ")
        ax[1].set_ylabel(" ")
        smallcoords(ax[1], 10, hx="z", vx="x", hc="b", vc="r")

        # z-y
        center_axis(ax[2])
        ax[2].set_xlabel(" ")
        ax[2].set_ylabel(" ")
        smallcoords(ax[2], 10, hx="z", vx="y", hc="b", vc="lime")

        # get timestamp
        now = datetime.now()
        timestamp = now.strftime("%m_%d_%Y_%I_%M_%S_%p")

        # save figure
        if savefig == True:
            fname = "TRACING_PLOT_2D_" + "d_" + timestamp
            save_figure(floc, fname)


def tracingplot_3d(rdat, ax=None, savefig=True, floc="../bin/"):

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection="3d")

    # radius of earth [m]
    RE = 6371000

    # scale coordinates in units [RE]
    rscale = rdat / RE

    # plot dipole field data
    ax.plot3D(rscale[:, 0], rscale[:, 1], rscale[:, 2], "k", linewidth=0.5)

    # plot starting point
    ax.plot(
        rscale[0][0],
        rscale[0][1],
        rscale[0][2],
        "r",
        marker="*",
        markersize=10,
    )

    # plot 3D earth
    draw_earth3D(ax, ecolor=nh_blue, rad=1)

    # graph correction, keeps origin (0,0,0) at center
    ranges = []
    ranges.append(math.ceil(max(abs(rscale[:, 0]))))
    ranges.append(math.ceil(min(abs(rscale[:, 0]))))
    ranges.append(math.ceil(max(abs(rscale[:, 1]))))
    ranges.append(math.ceil(min(abs(rscale[:, 1]))))
    ranges.append(math.ceil(max(abs(rscale[:, 2]))))
    ranges.append(math.ceil(min(abs(rscale[:, 2]))))

    axrng = max(ranges)

    ax.set_xlim([-axrng, axrng])
    ax.set_ylim([-axrng, axrng])
    ax.set_zlim([-axrng, axrng])

    # plot attributes
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("$x$ [$R_E$]")
    ax.set_ylabel("$y$ [$R_E$]")
    ax.set_zlabel("$z$ [$R_E$]")

    # Turn off tick labels
    ax.set_zticklabels([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # color labels
    ax.xaxis.label.set_color("r")
    ax.yaxis.label.set_color("lime")
    ax.zaxis.label.set_color("b")

    # get timestamp
    now = datetime.now()
    timestamp = now.strftime("%m_%d_%Y_%I_%M_%S_%p")

    # save figure
    if savefig == True:
        fname = "TRACING_PLOT_3D_" + "d_" + timestamp
        save_figure(floc, fname)
