from geopack import geopack
import xarray as xr
import argparse
import numpy as np
from tracing_tools import get_g_params, setup_fig, predict_b_gsm
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
    example_point_gsm = np.array([7, 7, 7])
    example_v_drift = data['V_drift_GSE'].values[37]
    example_efield = data['E_EDI'].values[37]
    # END EXAMPLES

    # part 2
    location_gse = example_point_gse
    time = example_time
    efield = example_efield

    # convert gse to gsm coords here!
    # location_gsm = hg.gse2gsm(location_gse)
    time_s = data['time'].values[37:39]
    gsm_matrix = hg.gse2gsm(np.array(time_s))[0]
    location_gsm = gsm_matrix.apply(location_gse)

    # part 4
    v_drift = example_v_drift

    # part 3
    # this is only needed for some setup stuff, geopack uses global variables which makes using it kind of confusing
    b_gsm=predict_b_gsm(time, location_gsm, v_drift)

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
    location_gsm_before = gsm_matrix.apply(location_gse_before)
    location_gsm_after = gsm_matrix.apply(location_gse_after)
    # NOT FOR REAL USE. ONLY UNTIL GSE2GSM WORKS
    # location_gsm_before = location_gse_before
    # location_gsm_after = location_gse_after


    # part 7
    pm = get_g_params(time)

    # location_before, location, and location_after are each a field line

    print('starting point for field line 0:', location_gsm)
    print('starting point for field line -1:', location_gsm_before)
    print('starting point for field line 1:', location_gsm_after)

    # is it because this isn't x/y/z? Because -21 and -45 seem like a lot

    # for r(t-dt)
    if location_gsm_before[2] <= 0:
        #x, y, z are all the last point in xx0, yy0, and zz0. xx,yy,zz contain every location over the line that is traced
        x0, y0, z0, xx0, yy0, zz0 = geopack.trace(location_gsm_before[0], location_gsm_before[1], location_gsm_before[2], dir=1, rlim=31, r0=.99999,
                                        parmod=pm, exname='t01', inname='dipole', maxloop=1000)
    else:
        # x, y, z are all the last point in xx0, yy0, and zz0. xx,yy,zz contain every location over the line that is traced
        x0, y0, z0, xx0, yy0, zz0 = geopack.trace(location_gsm_before[0], location_gsm_before[1], location_gsm_before[2], dir=-1, rlim=31, r0=.99999,
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

    # example plot, where x is the cartesian x and z is the cartesian z. May be incorrect, but tbd
    # ax=setup_fig()
    # ax.plot(xx0, zz0)
    # ax.plot(xx1, zz1)
    # ax.plot(xx2, zz2)
    # ax.set_title('Field Line Tracing for line -1 (blue), 0 (orange), and 1 (green?)')
    # plt.show()

    # polar cap location is the end point of the tracing (assuming the tracing converged correctly)
    polar_cap_0 = np.array([x0, y0, z0])
    polar_cap_1 = np.array([x1, y1, z1])
    polar_cap_2 = np.array([x2, y2, z2])

    # check if tracing converged correctly

    # magnetic equator location is where z=0 (doesn't happen for all tracings, so should probably find one where it does)
    # do we need the other location parts? probably
    mag_equator_ind_0 = np.argmin(np.abs(zz0))
    mag_equator_ind_1 = np.argmin(np.abs(zz1))
    mag_equator_ind_2 = np.argmin(np.abs(zz2))

    mag_equator_0 = np.array([xx0[mag_equator_ind_0], yy0[mag_equator_ind_0], zz0[mag_equator_ind_0]])
    mag_equator_1 = np.array([xx1[mag_equator_ind_1], yy1[mag_equator_ind_1], zz1[mag_equator_ind_1]])
    mag_equator_2 = np.array([xx2[mag_equator_ind_2], yy2[mag_equator_ind_2], zz2[mag_equator_ind_2]])

    # pick a better number than 1, just an example for now
    if mag_equator_0[2] > 1:
        print('field line -1 does not cross the magnetic equator')
    if mag_equator_1[2] > 1:
        print('field line 0 does not cross the magnetic equator')
    if mag_equator_2[2] > 1:
        print('field line 1 does not cross the magnetic equator')

    # part 8
    # V=E_x*Δx + E_y*Δy + E_z*Δz
    # we should do this in gsm, im pretty sure
    V = efield[0]*np.abs(location_gsm_after[0] - location_gsm_before[0]) + efield[1]*np.abs(location_gsm_after[1] - location_gsm_before[1]) + efield[2]*np.abs(location_gsm_after[2] - location_gsm_before[2])
    print('Potential:', V)

    # part 9




if __name__ == '__main__':
    main()