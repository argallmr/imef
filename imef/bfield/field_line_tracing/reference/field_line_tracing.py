# Following https://github.com/tsssss/geopack/blob/master/notebooks/Field%20Line%20Trace%20Demo.ipynb
# parmod parameters (incl. G1 and G2) taken from https://rbsp-ect.newmexicoconsortium.org/data_pub/QinDenton/

from matplotlib import pyplot as plt
from matplotlib.patches import Wedge, Circle
import numpy as np
from geopack import geopack

# Pulling THEMIS data from CDAWeb via the cdasws python library
from cdasws import CdasWs
cdas = CdasWs()

import pandas as pd
from datetime import timezone
from imef.bfield.field_line_tracing.tracing_tools import get_g_params


def dual_half_circle(center=(0, 0), radius=1, angle=90, ax=None, colors=('w', 'k', 'k'),
                     **kwargs):
    """
    Add two half circles to the axes *ax* (or the current axes) with the
    specified facecolors *colors* rotated at *angle* (in degrees).
    """
    if ax is None:
        ax = plt.gca()
    theta1, theta2 = angle, angle + 180
    # w1 = Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
    # w2 = Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)

    w1 = Wedge(center, radius, theta1, theta2, fc=colors[1], **kwargs)
    w2 = Wedge(center, radius, theta2, theta1, fc=colors[0], **kwargs)

    cr = Circle(center, radius, fc=colors[2], fill=False, **kwargs)
    for wedge in [w1, w2, cr]:
        ax.add_artist(wedge)
    return [w1, w2, cr]

def setup_fig(xlim=(10, -30), ylim=(-20, 20), xlabel='X GSM [Re]', ylabel='Z GSM [Re]'):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.axvline(0, ls=':', color='k')
    ax.axhline(0, ls=':', color='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_aspect('equal')
    w1, w2, cr = dual_half_circle(ax=ax)

    return ax

# Extract g parameters from file for "parmod" array
# (will still need to append recalc(ut) and sat gsm_xyz to resulting glist for use in trace())
# Need to find which entry in the list produced by read_qindenton_json
# corresponds to the correct timestamp
# Helper function to find the right dictionary for the given timestamp

# Converts timestamp to seconds since the unix epoch (in UTC)
# assumes dt is a datetime object
def get_epoch(dt):
    seconds = dt.replace(tzinfo=timezone.utc).timestamp()
    return seconds

# compress datetime64[s] to a string for saving files ([s] indicates precision to sec)
def datetimestr(map_time):
    full = str(map_time)
    dtstr = full[0:4]+full[5:7]+full[8:10]+full[11:13]+full[14:16]
    return dtstr #'YYYYMMDDHHMM'

# get THEMIS satellite coordinates from CDAWeb
# assumes datetime64[s] for timestamp input
def get_th_xyz(th_sat, timestamp):
    if th_sat in ('a', 'd', 'e'):
        satstr = 'TH' + th_sat.upper() + '_OR_SSC' #cdaWs expects 'THX_OR_SSC'
        #replace ' ' with 'T' in 'yyyy-mm-dd hh:mm:ss' of str(timestamp) and append 'Z' to end
        timestamp = str(timestamp).replace(' ', 'T')+'Z'
        xyz = cdas.get_data(satstr, ['XYZ_GSM'], timestamp, timestamp)[1]['XYZ_GSM']
        ### PROBLEM: THA_OR_SSC on CDAWeb for 2014-08-XX gives duplicate timestamps & coords.
        ### Fix: check that get_data() returns a 1D array.
        if(len(np.shape(xyz))) != 1: xyz = xyz[0]
        return xyz
    print('Must give char (a, d, e) for th_sat')
    return

# Return index where field line goes closest to z=0 and its value
def find_nearest_z(z_arr, value):
    array = np.asarray(z_arr)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def sketch_field(th_sat, test_time, ax=None, **plot_kwargs):
    # get smushed string form of test_time (used in get_g_file(), get_g_params())
    timestr = datetimestr(test_time)
    # get THEMIS coordinates
    x_gsm, y_gsm, z_gsm = get_th_xyz(th_sat, test_time)
    # Calculate dipole tilt angle
    ut = get_epoch(test_time)
    ps = geopack.recalc(ut)
    # Assemble parmod array
    pm = get_g_params(timestr) + [ps, x_gsm, y_gsm, z_gsm]

    # Calculate field line (both directions)
    x, y, z, xx, yy, zz = geopack.trace(x_gsm, y_gsm, z_gsm, dir=1, rlim=31, r0=.99999,
                                        parmod=pm, exname='t01', inname='igrf', maxloop=1000)

    # x2, y2, z2, xx2, yy2, zz2 = geopack.trace(x_gsm, y_gsm, z_gsm, dir=-1, rlim=31, r0=.99999,
    #                                           parmod=pm, exname='t01', inname='igrf', maxloop=1000)

    # xx = np.concatenate((np.flip(xx2), xx))
    # yy = np.concatenate((np.flip(yy2), yy))
    # zz = np.concatenate((np.flip(zz2), zz))
    # Check that field lines start and terminate at Earth
    if (abs(xx[0]) > 1 or abs(xx[-1]) > 1):
        print('Field line failed to terminate: TH' + th_sat.upper() + ': ' + str(test_time))

    # Plot figure
    if ax is None: ax = plt.gca()
    ax.plot(xx, zz, **plot_kwargs)
    ax.scatter(x_gsm, z_gsm, s=80, marker=(5, 2), label='th' + th_sat, zorder=2.5, **plot_kwargs)
    plt.title('T01 Test ' + str(test_time))
    mindex, z_min = find_nearest_z(zz, 0)
    ax.axvline(xx[mindex], ls=':', **plot_kwargs)  # color='r')
    # plt.show() #redundant; called automatically by matplotlib inline
    return (ax)

def main():
    map_time = pd.to_datetime('2014-08-05 11:45:00')  # '2014-08-05 11:45:00') 2010-04-04 11:45:00')
    test_ax = setup_fig()
    sketch_field('a', map_time, c='r')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()