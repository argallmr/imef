import numpy as np
import datetime as dt
from scipy.stats import binned_statistic_2d
import xarray as xr
from pymms.sdc import mrmms_sdc_api as api
import data_manipulation as dm
from pymms.data import edi, util, fgm
import argparse
import plot_nc_data as xrplot

# For debugging purposes
# np.set_printoptions(threshold=np.inf)


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
    E_sc = 1e-3 * np.cross(mec_data['V_sc'][:, :3], fgm_data['B'][:, :3])

    # Make into a DataArray to subtract the data easier
    E_sc = xr.DataArray(E_sc,
                        dims=['time', 'E_index'],
                        coords={'time': edi_data['time'],
                                'E_index': ['Ex', 'Ey', 'Ez']},
                        name='E_sc')

    # Remove E_sc from the measured electric field
    edi_data['E_GSE'] = edi_data['E_GSE'] - E_sc

    return edi_data


def remove_corot_efield(edi_data, mec_data, RE):
    E_corot = (-92100 * RE / np.linalg.norm(mec_data['R_sc'], ord=2,
                                            axis=mec_data['R_sc'].get_axis_num('R_sc_index')) ** 2)

    E_corot = xr.DataArray(E_corot, dims='time', coords={'time': mec_data['time']}, name='E_corot')

    edi_data = edi_data - E_corot

    return edi_data


def get_binned_statistics(edi_data, mec_data, nL, nMLT, L_range, MLT_range):
    # Count returns the amount of data points that fell in each bin
    # x_edge and y_edge represent the start and end of each of the bins in terms of L and MLT
    # binnum returns the bin number given to each data point in the dataset
    # values is not called when using statistic='count', but is still required.
    # Since E_GSE is in edi_data whether or not the user wants polar, it will work either way.
    count, x_edge, y_edge, binnum = binned_statistic_2d(x=mec_data['L'],
                                                        y=mec_data['MLT'],
                                                        values=edi_data['E_GSE'].loc[:, 'Ex'],
                                                        statistic='count',
                                                        bins=[nL, nMLT],
                                                        range=[L_range, MLT_range])

    return count, x_edge, y_edge, binnum


def create_imef_data(L, MLT, count, nL, nMLT, polar):
    # Creating an empty Dataset where the averaged data values will go
    if polar:
        L2, MLT2 = xr.broadcast(L, MLT)
        L2 = L2.rename({'L': 'iL', 'MLT': 'iMLT'})
        MLT2 = MLT2.rename({'L': 'iL', 'MLT': 'iMLT'})
        imef_data = xr.Dataset(coords={'L': L2, 'MLT': MLT2, 'polar': ['r', 'phi']})

        imef_data['count'] = xr.DataArray(count, dims=['iL', 'iMLT'], coords={'L': L2, 'MLT': MLT2})
        imef_data['E_mean'] = xr.DataArray(np.zeros((nL, nMLT, 2)), dims=['iL', 'iMLT', 'polar'],
                                           coords={'L': L2, 'MLT': MLT2})
        imef_data['E_std'] = xr.DataArray(np.zeros((nL, nMLT, 2)), dims=['iL', 'iMLT', 'polar'],
                                          coords={'L': L2, 'MLT': MLT2})
    else:
        L2, MLT2 = xr.broadcast(L, MLT)
        L2 = L2.rename({'L': 'iL', 'MLT': 'iMLT'})
        MLT2 = MLT2.rename({'L': 'iL', 'MLT': 'iMLT'})
        imef_data = xr.Dataset(coords={'L': L2, 'MLT': MLT2, 'cartesian': ['x', 'y', 'z']})

        imef_data['count'] = xr.DataArray(count, dims=['iL', 'iMLT'], coords={'L': L2, 'MLT': MLT2})
        imef_data['E_mean'] = xr.DataArray(np.zeros((nL, nMLT, 3)), dims=['iL', 'iMLT', 'cartesian'],
                                           coords={'L': L2, 'MLT': MLT2})
        imef_data['E_std'] = xr.DataArray(np.zeros((nL, nMLT, 3)), dims=['iL', 'iMLT', 'cartesian'],
                                          coords={'L': L2, 'MLT': MLT2})

    return imef_data


def bin_data(imef_data, edi_data, nL, nMLT, binnum, dL, dMLT, created_file, filename, polar):
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

        if polar:
            imef_data['E_mean'].loc[ir, ic, :] = edi_data['E_polar'][bool_idx, :].mean(dim='time')
            imef_data['E_std'].loc[ir, ic, :] = edi_data['E_polar'][bool_idx, :].std(dim='time')
        else:
            imef_data['E_mean'].loc[ir, ic, :] = edi_data['E_GSE'][bool_idx, :].mean(dim='time')
            imef_data['E_std'].loc[ir, ic, :] = edi_data['E_GSE'][bool_idx, :].std(dim='time')

    imef_data['L'] = imef_data['L'] + dL / 2
    imef_data['MLT'] = imef_data['MLT'] + dMLT / 2

    if not created_file:
        # If this is the first run, create a file (or overwrite any existing file) with the name defined by filename
        imef_data.to_netcdf(filename)
    else:
        # If this is not the first run, average the data in the file with the new imef_data
        # And export the new averaged data to the existing file
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
    average_std = (imef_data['count'] * (imef_data['E_std'] ** 2 + (imef_data['E_mean'] - average_mean) ** 2) +
                   file_data['count'] * (
                           file_data['E_std'] ** 2 + (file_data['E_mean'] - average_mean) ** 2)) / (
                          imef_data['count'] + file_data['count'])

    average_std = average_std.fillna(0)

    # Place the newly found averages to imef_data.
    # This could also just be made into a new Dataset, but it's easier this way.
    imef_data['count'].values = file_data['count'].values + imef_data['count'].values
    imef_data['E_mean'].values = average_mean.values
    imef_data['E_std'].values = average_std.values

    file_data.close()

    return imef_data


def main():
    # Collect Arguments
    parser = argparse.ArgumentParser(
        description='Download EDI data, average and bin the data by distance and orientation (L and MLT), '
                    'and store the data into a netCDF (.nc) file'
    )

    parser.add_argument('sc', type=str, help='Spacecraft Identifier')

    parser.add_argument('mode', type=str, help='Data rate mode')

    parser.add_argument('level', type=str, help='Data level')

    parser.add_argument('start_date', type=str, help='Start date of the data interval: ' '"YYYY-MM-DDTHH:MM:SS""')

    parser.add_argument('end_date', type=str, help='End date of the data interval: ''"YYYY-MM-DDTHH:MM:SS""')

    parser.add_argument('filename', type=str, help='Output file name')

    parser.add_argument('-n', '--no-show', help='Do not show the plot.', action='store_true')

    # If the polar plot is updated to spherical, update this note (and maybe change -p to -s)
    parser.add_argument('-p', '--polar', help='Convert the electric field values to polar', action='store_true')

    args = parser.parse_args()

    # Set up variables
    RE = 6371  # km. This is the conversion from km to Earth radii
    sc = args.sc  # Chosen spacecraft
    mode = args.mode  # Chosen data type
    level = args.level  # Chosen level

    # Start and end dates
    t0 = dt.datetime.strptime(args.start_date, '%Y-%m-%dT%H:%M:%S')
    t1 = dt.datetime.strptime(args.end_date, '%Y-%m-%dT%H:%M:%S')

    # Finds each individual orbit within that time frame and sorts them into a dictionary
    orbits = api.mission_events('orbit', t0, t1, sc)

    # Ranges for L and MLT values to be binned for
    L_range = (4, 10)  # RE
    MLT_range = (0, 24)  # Hours

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
    filename = args.filename

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

            # There are times where x, y, and z are copied, but the corresponding values are not, resulting in 6 coordinates
            # This throws an error when trying to computer the cross product in remove_spacecraft_efield, so raise an error here if this happens
            # Only seems to happen on one day, 12/18/16
            if len(mec_data["V_sc_index"]) != 3:
                raise ValueError("There should be 3 coordinates in V_sc_index")

        except Exception as ex:
            # Download will return an error when there is no data in the files that are being read.
            # Catch the error and print it out
            print('Failed: ', ex)
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

                # Remove the corotation electric field
                edi_data = remove_corot_efield(edi_data, mec_data, RE)

                if args.polar:
                    # Convert MEC and EDI data to polar coordinates
                    # Factor converts the MEC data from kilometers to RE
                    mec_data['r_polar'] = dm.cart2polar(mec_data['R_sc'], factor=1/RE)
                    edi_data['E_polar'] = (dm.rot2polar(edi_data['E_GSE'], mec_data['r_polar'], 'E_index')
                                           .assign_coords({'polar': ['r', 'phi']})
                                           )

                # Prepare to average and bin the data
                count, x_edge, y_edge, binnum = get_binned_statistics(edi_data, mec_data, nL, nMLT,
                                                                      L_range,
                                                                      MLT_range)

                # 2D spacial coordinates
                imef_data = create_imef_data(L, MLT, count, nL, nMLT, args.polar)

                # Average and bin the data into a file called filename
                imef_data = bin_data(imef_data, edi_data, nL, nMLT, binnum, dL, dMLT, created_file, filename,
                                     args.polar)

                # If this is the first run, let the program know that the file has been created
                # This is done so that any existing file with the same name as filename is overwritten,
                # and the new data is not averaged into any existing data from previous runs
                created_file = True

    # Plot the data, unless specified otherwise
    if not args.no_show:
        # If the user chose to plot edi data in polar coordinates, do so. Otherwise plot in cartesian
        if args.polar:
            xrplot.plot_polar_data(nL, nMLT, imef_data)
        else:
            xrplot.plot_cartesian_data(nL, nMLT, imef_data)


if __name__ == '__main__':
    main()
