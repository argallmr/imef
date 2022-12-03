from geopack import geopack
import xarray as xr
import argparse
import numpy as np
from tracing_tools import get_g_params, setup_fig
import hapgood as hg
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description='INSERT'
    )

    parser.add_argument('input_filename', type=str,
                        help='File name(s) of the data created by sample_data.py. Do not include file extension')

    args=parser.parse_args()

    data = xr.open_dataset(args.input_filename+'.nc')

    # EXAMPLES FOR NOW. GET REPLACED FOR FULL IMPLEMENTATION
    example_point_gse = np.array((data['L'].values[37], data['MLT'].values[37], data['MLAT'].values[37]))
    example_time = data['time'].values[37]
    example_point_gsm = np.array([10, 10, 10])
    example_v_drift = data['V_drift_GSE'].values[37]
    # END EXAMPLES

    # part 2
    location_gse = example_point_gse
    time = example_time

    # convert gse to gsm coords here!
    # location_gsm = hg.gse2gsm(location_gse)
    location_gsm = example_point_gsm

    # part 4
    v_drift = example_v_drift

    # part 3
    # looks like this isn't actually necessary, as all the predicting needed is done in the trace function
    # b_gsm=predict_b_gsm(time, location_gsm, v_drift)

    # part 5
    # just for reference, it is 5 seconds
    # sample_rate = dt.timedelta(seconds=5)

    # part 6
    #r(t-dt)=r-v*dt
    #r(t+dt)=r+v*dt
    # maybe do in gsm tbd
    # location data in RE, v_drift im Km/s, time in seconds. 1RE = 6378.12 km
    v_drift_right_units = np.array([v_drift[0]*5/6378.12, v_drift[1], v_drift[2]])
    location_gse_before = location_gse -v_drift_right_units
    location_gse_after = location_gse +v_drift_right_units

    # convert gse to gsm, as geopack can only use gsm or gsw
    # location_gsm_before = hg.gse2gsm(location_gse_before)
    # location_gsm_after = hg.gse2gsm(location_gse_after)
    # NOT FOR REAL USE. ONLY UNTIL GSE2GSM WORKS
    location_gsm_before = location_gse_before
    location_gsm_after = location_gse_after


    # part 7
    pm = get_g_params(time)

    # for r(t-dt)
    if location_gsm_before[2] <= 0:
        #x, y, z are all the last point in xx0, yy0, and zz0. xx,yy,zz contain every location over the line that is traced
        x0, y0, z0, xx0, yy0, zz0 = geopack.trace(location_gsm_before[0], location_gsm_before[1], location_gsm_before[2], dir=1, rlim=31, r0=.99999,
                                        parmod=pm, exname='t01', inname='dipole', maxloop=1000)
    else:
        # x, y, z are all the last point in xx0, yy0, and zz0. xx,yy,zz contain every location over the line that is traced
        x0, y0, z0, xx0, yy0, zz0 = geopack.trace(location_gsm_before[0], location_gsm_before[1],
                                                  location_gsm_before[2], dir=-1, rlim=31, r0=.99999,
                                                  parmod=pm, exname='t01', inname='dipole', maxloop=1000)
    # for r(t)
    if location_gsm[2] <= 0:
        # x, y, z are all the last point in xx0, yy0, and zz0. xx,yy,zz contain every location over the line that is traced
        x1, y1, z1, xx1, yy1, zz1 = geopack.trace(location_gsm[0], location_gsm[1], location_gsm[2], dir=1, rlim=31, r0=.99999,
                                                  parmod=pm, exname='t01', inname='dipole', maxloop=1000)
    else:
        # x, y, z are all the last point in xx0, yy0, and zz0. xx,yy,zz contain every location over the line that is traced
        x1, y1, z1, xx1, yy1, zz1 = geopack.trace(location_gsm[0], location_gsm[1], location_gsm[2], dir=-1, rlim=31, r0=.99999,
                                                  parmod=pm, exname='t01', inname='dipole', maxloop=1000)
    # for r(t+dt)
    if location_gsm_after[2] <= 0:
        # x, y, z are all the last point in xx0, yy0, and zz0. xx,yy,zz contain every location over the line that is traced
        x2, y2, z2, xx2, yy2, zz2 = geopack.trace(location_gsm_after[0], location_gsm_after[1], location_gsm_after[2], dir=1, rlim=31, r0=.99999,
                                                  parmod=pm, exname='t01', inname='dipole', maxloop=1000)
    else:
        # x, y, z are all the last point in xx0, yy0, and zz0. xx,yy,zz contain every location over the line that is traced
        x2, y2, z2, xx2, yy2, zz2 = geopack.trace(location_gsm_after[0], location_gsm_after[1], location_gsm_after[2], dir=-1, rlim=31, r0=.99999,
                                                  parmod=pm, exname='t01', inname='dipole', maxloop=1000)

    # for some reason xx2 and zz2 only have 1 entry. tbd why
    ax=setup_fig()
    ax.plot(xx0, zz0)
    ax.plot(xx1, zz1)
    ax.plot(xx2, zz2)
    plt.show()

    # part 8

    # part 9




if __name__ == '__main__':
    main()