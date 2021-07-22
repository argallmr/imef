import numpy as np
import datetime as dt
from scipy.stats import binned_statistic_2d
import xarray as xr
from data_manipulation import rot2polar, cart2polar, remove_corot_efield, remove_spacecraft_efield
import argparse
import plot_nc_data as xrplot
from download_data import get_fgm_data, get_edi_data, get_mec_data

# For debugging purposes
# np.set_printoptions(threshold=np.inf)


def prep_and_store_data(edi_data, fgm_data, mec_data, filename, polar, created_file, nL, nMLT, L_range, MLT_range, L, MLT, dL, dMLT):
    RE = 6371  # km. This is the conversion from km to Earth radii

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

        if polar:
            # Convert MEC and EDI data to polar coordinates
            # Factor converts the MEC data from kilometers to RE
            # Instead of positive x facing the right, it is facing the left in MEC data. So we have to shift the angle around 180 degrees so the directions match up
            mec_data['r_polar'] = cart2polar(mec_data['R_sc'], factor=1 / RE, shift=np.pi)
            edi_data['E_polar'] = (rot2polar(edi_data['E_GSE'], mec_data['r_polar'], 'E_index').assign_coords({'polar': ['r', 'phi']}))

        # Prepare to average and bin the data
        count, x_edge, y_edge, binnum = get_binned_statistics(edi_data, mec_data, nL, nMLT, L_range, MLT_range)

        # Create the empty dataset
        imef_data = create_imef_data(L, MLT, count, nL, nMLT, polar)

        # Average and bin the data into a file called filename
        imef_data = bin_data(imef_data, edi_data, nL, nMLT, binnum, dL, dMLT, created_file, filename,
                             polar)

        # If this is the first run, let the program know that the file has been created
        # This is done so that any existing file with the same name as filename is overwritten,
        # and the new data is not averaged into any existing data from previous runs
        created_file = True

        return imef_data, created_file


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

    print(binnum)
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

    # Each bin goes from x to x+dL, but the index associating those values only starts at the beginning of the bin, which is misleading
    # Change the index to be in the middle of the bin
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
                   file_data['count'] * (file_data['E_std'] ** 2 + (file_data['E_mean'] - average_mean) ** 2)) / \
                  (imef_data['count'] + file_data['count'])

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

    parser.add_argument('-e', '--exists', help='The output file already exists, merge the data across the new dates with the existing data', action='store_true')

    # If the polar plot is updated to spherical, update this note (and maybe change -p to -s)
    parser.add_argument('-p', '--polar', help='Convert the electric field values to polar', action='store_true')

    args = parser.parse_args()

    # Set up variables
    sc = args.sc  # Chosen spacecraft
    mode = args.mode  # Chosen data type
    level = args.level  # Chosen level

    # Start and end dates
    t0 = dt.datetime.strptime(args.start_date, '%Y-%m-%dT%H:%M:%S')
    t1 = dt.datetime.strptime(args.end_date, '%Y-%m-%dT%H:%M:%S')

    # The name of the file where the data will be stored
    filename = args.filename
    # Whether the user wants the data in polar coordinates or not
    polar = args.polar

    # A datetime number to increment the dates by one day, but without extending into the next day
    one_day = dt.timedelta(days=1) - dt.timedelta(microseconds=1)

    # Ranges for L and MLT values to be binned for
    L_range = (0, 25)  # RE
    MLT_range = (0, 24)  # Hours

    # Size of the L and MLT bins
    dL = 1  # RE
    dMLT = 1  # Hours

    # DataArrays with desired L and MLT bins
    L = xr.DataArray(np.arange(L_range[0], L_range[1], dL), dims='L')
    MLT = xr.DataArray(np.arange(MLT_range[0], MLT_range[1], dMLT), dims='MLT')

    # Number of points in each coordinate
    nL = len(L)
    nMLT = len(MLT)

    # Boolean containing whether the file has been created
    if args.exists:
        created_file = True
    else:
        created_file = False

    # Download and process the data separately for each individual orbit,
    # So loop through every value in orbits
    while t0 < t1:
        # Assign the start time
        ti = t0

        # Determine the time difference between ti and midnight. Only will be a non-zero number on the first run through the loop
        timediff = dt.datetime.combine(dt.date.min, ti.time()) - dt.datetime.min

        # Assign the end date. timediff is used so that data is only downloaded from 1 day per run through the loop, which prevents bugs from appearing
        te = ti + one_day - timediff

        # If te were to extend past the desired end date, make te the desired end date
        if te > t1:
            te = t1

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
            imef_data, created_file = prep_and_store_data(edi_data, fgm_data, mec_data, filename, polar, created_file, nL, nMLT, L_range, MLT_range, L, MLT, dL, dMLT)

        # Increment the start day by an entire day, so that the next run in the loop starts on the next day
        t0 = ti + dt.timedelta(days=1) - timediff

    # Plot the data, unless specified otherwise
    if not args.no_show:
        # If the user chose to store edi data in polar coordinates, plot that data. Otherwise plot in cartesian
        if args.polar:
            xrplot.plot_polar_data(nL, nMLT, imef_data)
        else:
            xrplot.plot_cartesian_data(nL, nMLT, imef_data)


if __name__ == '__main__':
    main()
