import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic_2d
import xarray as xr
from pymms.sdc import mrmms_sdc_api as api
import data_manipulation as dm
from pymms.data import edi, util, fgm


def get_edi_data(sc, mode, level, ti, te):
    tm_vname = '_'.join((sc, 'edi', 't', 'delta', 'minus', mode, level))

    # Get EDI data
    edi_data = edi.load_data(sc, mode, level,
                             optdesc='efield', start_date=ti, end_date=te)

    # Timestamps begin on 0's and 5's and span 5 seconds. The timestamp is at
    # the weighted mean of all beam hits. To get the beginning of the timestamp,
    # subtract the time's DELTA_MINUS. But this is inaccurate by a few nanoseconds
    # so we have to round to the nearest second.
    edi_time = edi_data['Epoch'] - edi_data[tm_vname].astype('timedelta64[ns]')
    edi_time = [(t - tdelta)
                if tdelta.astype(int) < 5e8
                else (t + np.timedelta64(1, 's') - tdelta)
                for t, tdelta in zip(edi_time.data, edi_time.data - edi_time.data.astype('datetime64[s]'))
                ]

    # Replace the old time data with the corrected time data
    edi_data['Epoch'] = edi_time

    # Rename Epoch to time for consistency across all data files
    edi_data = edi_data.rename({'Epoch': 'time'})

    return edi_data


def get_fgm_data(sc, mode, ti, te):
    # Get FGM data
    fgm_data = fgm.load_data(sc=sc, mode=mode, start_date=ti, end_date=te)

    # Rename some variables
    fgm_data = fgm_data.rename({'B_GSE': 'B'})

    return fgm_data


def get_mec_data(sc, mode, level, ti, te):
    # The names of the variables that will be downloaded
    r_vname = '_'.join((sc, 'mec', 'r', 'gse'))
    v_vname = '_'.join((sc, 'mec', 'v', 'gse'))
    mlt_vname = '_'.join((sc, 'mec', 'mlt'))
    l_dip_vname = '_'.join((sc, 'mec', 'l', 'dipole'))

    # The names of the indices used for the radius (?) and velocity data
    r_lbl_vname = '_'.join((sc, 'mec', 'r', 'gse', 'label'))
    v_lbl_vname = '_'.join((sc, 'mec', 'v', 'gse', 'label'))

    # Get MEC data
    mec_data = util.load_data(sc, 'mec', mode, level,
                              optdesc='epht89d', start_date=ti, end_date=te,
                              variables=[r_vname, v_vname, mlt_vname, l_dip_vname])

    # Rename variables
    mec_data = mec_data.rename({r_vname: 'R_sc',
                                r_lbl_vname: 'R_sc_index',
                                v_vname: 'V_sc',
                                v_lbl_vname: 'V_sc_index',
                                mlt_vname: 'MLT',
                                l_dip_vname: 'L',
                                'Epoch': 'time',
                                })
    return mec_data


def remove_spacecraft_efield(edi_data, fgm_data, mec_data):
    # E = v x B, 1e-3 converts units to mV/m
    E_sc = 1e-3 * np.cross(mec_data['V_sc'], fgm_data['B'][:, :3])

    # Make into a DataArray to subtract the data easier
    E_sc = xr.DataArray(E_sc,
                        dims=['time', 'E_index'],
                        coords={'time': edi_data['time'],
                                'E_index': ['Ex', 'Ey', 'Ez']},
                        name='E_sc')

    # Remove E_sc from the measured electric field
    edi_data['E_GSM'] = edi_data['E_GSM'] - E_sc

    return edi_data


def get_binned_statistics(edi_data, mec_data, nL, nMLT, L_range, MLT_range):
    # Count returns the amount of data points that fell in each bin
    # x_edge and y_edge represent the start and end of each of the bins in terms of L and MLT
    # binnum returns the bin number given to each data point in the dataset
    count, x_edge, y_edge, binnum = binned_statistic_2d(x=mec_data['L'],
                                                        y=mec_data['MLT'],
                                                        values=edi_data['E_polar'].loc[:, 'r'],
                                                        statistic='count',
                                                        bins=[nL, nMLT],
                                                        range=[L_range, MLT_range])

    return count, x_edge, y_edge, binnum


def create_imef_data(L, MLT, count, nL, nMLT):
    # Creating an empty Dataset where the averaged data values will go
    L2, MLT2 = xr.broadcast(L, MLT)
    L2 = L2.rename({'L': 'iL', 'MLT': 'iMLT'})
    MLT2 = MLT2.rename({'L': 'iL', 'MLT': 'iMLT'})
    imef_data = xr.Dataset(coords={'L': L2, 'MLT': MLT2, 'polar': ['r', 'phi']})

    imef_data['count'] = xr.DataArray(count, dims=['iL', 'iMLT'], coords={'L': L2, 'MLT': MLT2})
    imef_data['E_mean'] = xr.DataArray(np.zeros((nL, nMLT, 2)), dims=['iL', 'iMLT', 'polar'],
                                       coords={'L': L2, 'MLT': MLT2})
    imef_data['E_std'] = xr.DataArray(np.zeros((nL, nMLT, 2)), dims=['iL', 'iMLT', 'polar'],
                                      coords={'L': L2, 'MLT': MLT2})

    return imef_data


def bin_data(imef_data, edi_data, nL, nMLT, binnum, created_file, filename):
    for ibin in range((nL + 2) * (nMLT + 2)):
        # `binned_statistic_2d` adds one bin before and one bin after the specified
        # bin `range` in each dimension. This means the number of bin specified by
        # `bins` is actually two less per dimension than the actual number of bins used.
        # These are the indices into the "N+2" grid
        icol = ibin % (nMLT + 2)
        irow = ibin // (nMLT + 2)
        bin = (icol + 1) + irow * (nMLT + 2)  # binned_statistic_2d bin number (1-based: ibin+1)
        if (irow == 0) | (irow > nL) | (icol == 0) | (icol > nMLT):
            continue

        # These are the indices into our data (skipping bins outside our data range)
        ir = irow - 1  # data row
        ic = icol - 1  # data column
        ib = ic + ir * (nMLT + 2)  # flattened data index

        # Do not do anything if the bin is empty
        #   - equivalently: if imef_data['count'][ir,ic] == 0:
        bool_idx = binnum == ibin
        if sum(bool_idx) == 0:
            continue

        imef_data['E_mean'].loc[ir, ic, :] = edi_data['E_polar'][bool_idx, :].mean(dim='time')
        imef_data['E_std'].loc[ir, ic, :] = edi_data['E_polar'][bool_idx, :].std(dim='time')

    if not created_file:
        # If this is the first run, create a file (or overwrite any existing file) called binned.nc
        imef_data.to_netcdf(filename)
    else:
        # If this is not the first run, average the data in the file binned.nc with the new imef_data
        # And export the new averaged data to binned.nc
        imef_data = average_data(imef_data, filename)
        imef_data.to_netcdf(filename)

    return imef_data


def average_data(imef_data, filename):
    # Open file
    file_data = xr.open_dataset(filename)

    # Calculate weighted mean of incoming and existing data
    average_mean = (imef_data['count'] * imef_data['E_mean'] + file_data['count'] * file_data['E_mean']) / (
                imef_data['count'] + file_data['count'])

    # For any bins that have 0 data points, the above divides by 0 and returns NaN. Change the NaN's to 0's
    average_mean = average_mean.fillna(0)

    # Calculate weighted standard deviation of incoming and existing data
    average_std = (imef_data['count'] * (imef_data['E_std'] ** 2 + (imef_data['E_mean'] - average_mean) ** 2) + file_data['count'] * (
            file_data['E_std'] ** 2 + (file_data['E_mean'] - average_mean) ** 2)) / (imef_data['count'] + file_data['count'])

    average_std = average_std.fillna(0)

    # Place the newly found averages to imef_data.
    # This could also just be made into a new Dataset, but it's easier this way.
    imef_data['count'].values = file_data['count'].values + imef_data['count'].values
    imef_data['E_mean'].values = average_mean.values
    imef_data['E_std'].values = average_std.values

    file_data.close()

    return imef_data


def draw_earth(ax):
    '''
    A handy function for drawing the Earth in a set of Polar Axes
    '''
    ax.fill_between(np.linspace(-np.pi / 2, np.pi / 2, 30), 0, np.ones(30), color='k')
    ax.plot(np.linspace(np.pi / 2, 3 * np.pi / 2, 30), np.ones(30), color='k')


def plot_data(nL, nMLT, dL, dMLT, imef_data):

    # Create a coordinate grid
    phi = (2 * np.pi * (imef_data['MLT'].values + dMLT/2) / 24).reshape(nL, nMLT)
    r = imef_data['L'].values.reshape(nL, nMLT) + dL/2
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

def main():
    # Set up variables
    RE = 6371  # km. This is the conversion from km to Earth radii
    sc = 'mms1'  # Chosen spacecraft
    mode = 'srvy'  # Chosen data type
    level = 'l2'  # Chosen level

    # Start and end dates
    t0 = dt.datetime(2015, 9, 10, 0, 0, 0)
    t1 = dt.datetime(2015, 9, 22, 0, 0, 0)

    # Finds each individual orbit within that time frame and sorts them into a dictionary
    orbits = api.mission_events('orbit', t0, t1, sc)

    # Ranges for L and MLT values to be binned for
    L_range = (0, 12) # RE
    MLT_range = (0, 24) # Hours

    # Size of the L and MLT bins
    dL = 1  # RE
    dMLT = 1  # Hours (Does this catch the last bin values?)

    # DataArrays with desired L and MLT bins
    L = xr.DataArray(np.arange(L_range[0], L_range[1], dL), dims='L')
    MLT = xr.DataArray(np.arange(MLT_range[0], MLT_range[1], dMLT), dims='MLT')

    # Number of points in each coordinate
    nL = len(L)
    nMLT = len(MLT)

    # Name of the file where the data will be stored
    filename = 'binned.nc'

    # Boolean containing whether the file has been created
    created_file = False

    # Download and process the data separately for each individual orbit,
    # So loop through every value in orbits
    for orbit_count in range(len(orbits['tstart'])):
        # Selects the start and end dates from the orbits dictionary
        ti = orbits['tstart'][orbit_count]
        te = orbits['tend'][orbit_count]
        print(ti, '%%', te)
        try:
            # Read EDI data
            edi_data = get_edi_data(sc, mode, level, ti, te)

            # Read FGM data
            fgm_data = get_fgm_data(sc, mode, ti, te)

            # Read MEC data
            mec_data = get_mec_data(sc, mode, level, ti, te)

        except Exception as ex:
            # Download will return an error when there is no data in the files that are being read.
            # Catch the error and print it out
            print('Failed because', ex)
        else:
            # Make sure that the data file is not empty
            if edi_data.time.size != 0:
                # EDI, FGM, and MEC data all have a different number of data points at different times.
                # EDI has the lowest amount of data points, so take all the data points at the times EDI has
                # and use those points for FGM and MEC data
                fgm_data = fgm_data.interp_like(edi_data)
                mec_data = mec_data.interp_like(edi_data)

                # The spacecraft creates its own electric field and must be removed from the total calculations
                edi_data = remove_spacecraft_efield(edi_data, fgm_data, mec_data)

                # Convert MEC and EDI data to polar coordinates
                # Factor converts the MEC data from kilometers to RE
                mec_data['r_polar'] = dm.cart2polar(mec_data['R_sc'], factor=RE)
                edi_data['E_polar'] = (dm.rot2polar(edi_data['E_GSE'], mec_data['r_polar'], 'E_index')
                                       .assign_coords({'polar': ['r', 'phi']})
                                       )

                # Prepare to average and bin the data
                count, x_edge, y_edge, binnum = get_binned_statistics(edi_data, mec_data, nL, nMLT,
                                                                      L_range,
                                                                      MLT_range)

                # 2D spacial coordinates
                imef_data = create_imef_data(L, MLT, count, nL, nMLT)

                # Average and bin the data into binned.nc
                imef_data = bin_data(imef_data, edi_data, nL, nMLT, binnum, created_file, filename)

                # If this is the first run, let the program know that the file has been created
                # This is done so that any existing file called binned.nc is overwritten,
                # and the new data is not averaged into any existing data from previous runs
                created_file = True

    # Plot the data
    plot_data(nL, nMLT, dL, dMLT, imef_data)


if __name__ == '__main__':
    main()
