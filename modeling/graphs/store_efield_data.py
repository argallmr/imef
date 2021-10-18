import numpy as np
import datetime as dt
from scipy.stats import binned_statistic_2d
import xarray as xr
from data_manipulation import rot2polar, cart2polar, remove_corot_efield, remove_spacecraft_efield
import argparse
import plot_nc_data as xrplot
from storage_objects import LandMLT
from download_data import get_fgm_data, get_edi_data, get_mec_data, get_kp_data, get_IEF

# For debugging purposes
# np.set_printoptions(threshold=np.inf)

# Known Issue: Standard deviation does not work properly. It works over short intervals (ex: 9/10/15-9/15/15, 9/10/15-10/15/15), but over the whole 6 years numerous 0 and inf values appear
# I believe it is a small issue with an error not getting caught, or something similar. Possibly Fixed

# TO DO: Modify this, sample_data, and download_data to use the DownloadParameters container instead of using individual arguments
# Driving parameter and extra data work together. They don't. It gets separated but not actually ordered to bin. Maybe this is fixed?


def prep_and_store_data(edi_data, fgm_data, mec_data, filename, polar, created_file, L_and_MLT, ti, te, extra_data, driving_parameter):
    # EDI data must be prepared separately to remove unwanted parts of the data (and convert the data to polar if the user chose to do so).
    # Turns out the NaN's appear in interp_like. A couple values get trashed in there. Not a big deal since bin handles the NaN's, but would be nice to remove.
    mec_data, data_to_bin = prep_data(edi_data, fgm_data, mec_data, polar, extra_data, driving_parameter, ti, te, L_and_MLT)

    # Prepare to average and bin the data.
    count, x_edge, y_edge, binnum = get_binned_statistics(data_to_bin[0][0], mec_data, L_and_MLT)

    # Create the empty dataset
    imef_data = create_imef_data(data_to_bin, L_and_MLT, count, polar)

    # Iterate through data_to_bin and bin all the data
    for data in data_to_bin:

        # Average and bin the data into a file called filename
        imef_data = bin_data(imef_data, data[0], data[1], L_and_MLT, binnum, created_file, filename)

        # Add the newly binned data to the existing data
        if data == data_to_bin[0]:
            all_binned_data = [imef_data]
        else:
            all_binned_data.append(imef_data)

    imef_data = xr.merge(all_binned_data)

    imef_data.to_netcdf(filename)

    # If this is the first run, let the program know that the file has been created
    # This is done so that any existing file with the same name as filename is overwritten,
    # and the new data is not averaged into any existing data from previous runs
    created_file = True

    return imef_data, created_file


def prep_data(edi_data, fgm_data, mec_data, polar, extra_data, driving_parameter, ti, te, L_and_MLT):
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

        # A dictionary containing the keys for extra data that is available, and the maximum value that the index will measure
        # As more options are created they will be added into here.
        # This calls all the functions inside when creating dict. Bad. Also not 100% on the max IEF value. It kinda corresponds to Matsui but not
        x = {'Kp': [get_kp_data(ti, te, edi_data['time'].values), 9],
             'IEF': [get_IEF(ti, te, edi_data['time'].values), 3]}

        # A 2D list that contains the data to be binned, along with the name that it will be given when binned.

        if polar:
            # Convert MEC and EDI data to polar coordinates
            # Factor converts the MEC data from kilometers to RE
            # Instead of positive x facing the right, it is facing the left in MEC data. So we have to shift the angle around 180 degrees so the directions match up
            mec_data['r_polar'] = cart2polar(mec_data['R_sc'], factor=1 / RE, shift=np.pi)
            edi_data['E_GSE_polar'] = (rot2polar(edi_data['E_GSE'], mec_data['r_polar'], 'E_index').assign_coords({'polar': ['r', 'phi']}))
            edi_name='E_GSE_polar'
        else:
            edi_name='E_GSE'

        # This is a list that contains an xarray dataset with the data to be binned, the name of the variable to be binned,
        # and a boolean that states whether the data depends on cartesian/polar coordinates
        data_to_bin = [[edi_data, edi_name, True]]

        # If there is extra data, iterate through the given list of desired variables, download them, and add them to the list of data to bin
        extra_data_values = []
        if extra_data[0] != None:
            for variable in extra_data:
                more_data = x[variable]
                extra_data_values = extra_data_values.append(more_data)
                data_to_bin.append([more_data, variable, False]) #NOTE THAT THIS MAY NOT ALWAYS BE FALSE. IF THERE IS ONE THAT IS TRUE THIS NEEDS TO BE ADJUSTED

        # If there is a driving parameter, handle it and get the newly separated data
        if driving_parameter[0] != None:
            data_to_bin= handle_driving_parameter(driving_parameter, x, extra_data_values, extra_data, edi_data, edi_name, mec_data, data_to_bin, L_and_MLT)
    else:
        # There should be stuff in the file, so if there isn't don't do anything
        raise ValueError("Electric field data file is empty")

    return mec_data, data_to_bin


def handle_driving_parameter(driving_parameter, x, extra_data_values, extra_data_name, edi_data, edi_name, mec_data, data_to_bin, L_and_MLT):
    # NOTE: I think that this works with both extra data and driving parameter being used at the same time, but I can't test it so not positive

    # Download the driving parameter data
    driving_param_data, max_value = x[driving_parameter[0]]

    # Create one xarray object with all the data that is going to be binned
    total_data = xr.merge(extra_data_values)
    total_data = xr.merge([total_data, driving_param_data, edi_data])

    # Determine the step that the while loop will use
    step = max_value / int(driving_parameter[1])

    if extra_data_name[0]!=None:
        all_names = extra_data_name.append(edi_name)
    else:
        all_names = [edi_name]

    # Iterate through all the ranges of data that will be separated (counter -> counter+step)
    for name in all_names:
        # Create an empty list that will contain all the newly separated data
        # DO I NEED TO MOVE THE EMPTY LISTS OUTSIDE OF THE FOR LOOP? (for when driving and extra are both used)
        separated_data = []

        # There are two separate counters, create (or reset) them now
        counter = 0
        counter2 = 0

        while counter < max_value:
            # Create a new dataset that contains the data in the range that we want
            intermediate_step = total_data.where(total_data[driving_parameter[0]] <= counter + step, drop=True)
            imef_data = intermediate_step.where(total_data[driving_parameter[0]] > counter, drop=True)

            # Need to get the corresponding mec data for our new imef data
            if imef_data.time.size == 0:
                # Create an empty mec_data dataset (no time will be less than the epoch)
                new_mec_data = mec_data.where(mec_data.time < np.datetime64(0, 's'), drop=True)
            else:
                # Take the values that are in the imef_data
                new_mec_data = mec_data.interp_like(imef_data)

            new_name = name + '_' + driving_parameter[0] + '_' + str(counter) + '_to_' + str(counter + step)

            count, x_edge, y_edge, binnum = get_binned_statistics(imef_data, new_mec_data, L_and_MLT)

            imef_data[new_name+'_count'] = xr.DataArray(count, dims=['L', 'MLT'], coords={'L': L_and_MLT.L, 'MLT': L_and_MLT.MLT})
            # Just make a new list containing all of these, return, put into create_imef_data, and make there. Then should be good

            # Rename the data variable to include the range in the name
            imef_data = imef_data.rename({name: new_name})

            # Add to the list of datasets
            separated_data.append(imef_data)
            counter += step

        # Combine all the separated datasets into 1 dataset (Note this is so all the datasets have the same number of times, which prevents bugs in the binning phase)
        # Doing this results in a bunch of NaN values in the empty times that are undefined in the sectioned datasets
        test = xr.merge(separated_data)

        # Add the new data to the data_to_bin object, with the required other data (variable name and coordinate dependence)
        while counter2 < max_value:
            bin_name = name + '_' + driving_parameter[0] + '_' + str(counter2) + '_to_' + str(counter2 + step)
            data_to_bin.append([test, bin_name, True])
            counter2 += step

    return data_to_bin


def get_binned_statistics(data, mec_data, L_and_MLT):
    # Count returns the amount of data points that fell in each bin
    # x_edge and y_edge represent the start and end of each of the bins in terms of L and MLT
    # binnum returns the bin number given to each data point in the dataset
    # values is not called when using statistic='count', but is still required.
    # Since E_GSE is in edi_data whether or not the user wants polar, it will work either way.
    count, x_edge, y_edge, binnum = binned_statistic_2d(x=mec_data['L'],
                                                        y=mec_data['MLT'],
                                                        values=data['E_GSE'].loc[:, 'Ex'],
                                                        statistic='count',
                                                        bins=[L_and_MLT.nL, L_and_MLT.nMLT],
                                                        range=[L_and_MLT.L_range, L_and_MLT.MLT_range])

    return count, x_edge, y_edge, binnum


def create_imef_data(data, L_and_MLT, count, polar):
    # Creating an empty Dataset where the averaged EDI data values will go
    if polar:
        L2, MLT2 = xr.broadcast(L_and_MLT.L, L_and_MLT.MLT)
        L2 = L2.rename({'L': 'iL', 'MLT': 'iMLT'})
        MLT2 = MLT2.rename({'L': 'iL', 'MLT': 'iMLT'})
        imef_data = xr.Dataset(coords={'L': L2, 'MLT': MLT2, 'polar': ['r', 'phi']})

        imef_data['E_GSE_polar_count'] = xr.DataArray(count, dims=['iL', 'iMLT'], coords={'L': L2, 'MLT': MLT2})
        imef_data['E_GSE_polar_mean'] = xr.DataArray(np.zeros((L_and_MLT.nL, L_and_MLT.nMLT, 2)),
                                                     dims=['iL', 'iMLT', 'polar'],
                                                     coords={'L': L2, 'MLT': MLT2})
        imef_data['E_GSE_polar_std'] = xr.DataArray(np.zeros((L_and_MLT.nL, L_and_MLT.nMLT, 2)),
                                                    dims=['iL', 'iMLT', 'polar'],
                                                    coords={'L': L2, 'MLT': MLT2})
    else:
        L2, MLT2 = xr.broadcast(L_and_MLT.L, L_and_MLT.MLT)
        L2 = L2.rename({'L': 'iL', 'MLT': 'iMLT'})
        MLT2 = MLT2.rename({'L': 'iL', 'MLT': 'iMLT'})
        imef_data = xr.Dataset(coords={'L': L2, 'MLT': MLT2, 'cartesian': ['x', 'y', 'z']})

        imef_data['E_GSE_count'] = xr.DataArray(count, dims=['iL', 'iMLT'], coords={'L': L2, 'MLT': MLT2})
        imef_data['E_GSE_mean'] = xr.DataArray(np.zeros((L_and_MLT.nL, L_and_MLT.nMLT, 3)),
                                               dims=['iL', 'iMLT', 'cartesian'],
                                               coords={'L': L2, 'MLT': MLT2})
        imef_data['E_GSE_std'] = xr.DataArray(np.zeros((L_and_MLT.nL, L_and_MLT.nMLT, 3)),
                                              dims=['iL', 'iMLT', 'cartesian'],
                                              coords={'L': L2, 'MLT': MLT2})

    # Each bin goes from x to x+dL, but the index associating those values only starts at the beginning of the bin, which is misleading
    # Change the index to be in the middle of the bin
    imef_data['L'] = imef_data['L'] + L_and_MLT.dL / 2
    imef_data['MLT'] = imef_data['MLT'] + L_and_MLT.dMLT / 2

    # If there are other data points that need to be binned, create those data values here
    for index in range(1, len(data)):
        data_name = data[index][1]
        if data[index][2] == False:
            imef_data[data_name + '_count'] = xr.DataArray(data[index][0][data_name+'_count'], dims=['iL', 'iMLT'], coords={'L': L2, 'MLT': MLT2})
            imef_data[data_name + '_mean'] = xr.DataArray(np.zeros((L_and_MLT.nL, L_and_MLT.nMLT)), dims=['iL', 'iMLT'],
                                                          coords={'L': L2, 'MLT': MLT2})
            imef_data[data_name + '_std'] = xr.DataArray(np.zeros((L_and_MLT.nL, L_and_MLT.nMLT)), dims=['iL', 'iMLT'],
                                                         coords={'L': L2, 'MLT': MLT2})
        else:
            if polar:
                imef_data[data_name + '_count'] = xr.DataArray(data[index][0][data_name + '_count'],
                                                               dims=['iL', 'iMLT'], coords={'L': L2, 'MLT': MLT2})
                imef_data[data_name + '_mean'] = xr.DataArray(np.zeros((L_and_MLT.nL, L_and_MLT.nMLT, 2)), dims=['iL', 'iMLT', 'polar'],
                                                              coords={'L': L2, 'MLT': MLT2})
                imef_data[data_name + '_std'] = xr.DataArray(np.zeros((L_and_MLT.nL, L_and_MLT.nMLT, 2)), dims=['iL', 'iMLT', 'polar'],
                                                             coords={'L': L2, 'MLT': MLT2})
            else:
                imef_data[data_name + '_count'] = xr.DataArray(data[index][0][data_name + '_count'],
                                                               dims=['iL', 'iMLT'], coords={'L': L2, 'MLT': MLT2})
                imef_data[data_name + '_mean'] = xr.DataArray(np.zeros((L_and_MLT.nL, L_and_MLT.nMLT, 3)), dims=['iL', 'iMLT', 'cartesian'],
                                                            coords={'L': L2, 'MLT': MLT2})
                imef_data[data_name + '_std'] = xr.DataArray(np.zeros((L_and_MLT.nL, L_and_MLT.nMLT, 3)), dims=['iL', 'iMLT', 'cartesian'],
                                                             coords={'L': L2, 'MLT': MLT2})

    return imef_data


def bin_data(imef_data, data, data_name, L_and_MLT, binnum, created_file, filename):
    for ibin in range((L_and_MLT.nL + 2) * (L_and_MLT.nMLT + 2)):
        # `binned_statistic_2d` adds one bin before and one bin after the specified
        # bin `range` in each dimension. This means the number of bin specified by
        # `bins` is actually two less per dimension than the actual number of bins used.
        # These are the indices into the "N+2" grid
        icol = ibin % (L_and_MLT.nMLT + 2)
        irow = ibin // (L_and_MLT.nMLT + 2)
        bin = (icol + 1) + irow * (L_and_MLT.nMLT + 2)  # binned_statistic_2d bin number (1-based: ibin+1)
        if (irow == 0) | (irow > L_and_MLT.nL) | (icol == 0) | (icol > L_and_MLT.nMLT):
            continue

        # These are the indices into our data (skipping bins outside our data range)
        ir = irow - 1  # data row
        ic = icol - 1  # data column
        ib = ic + ir * (L_and_MLT.nMLT + 2)  # flattened data index

        # Do not do anything if the bin is empty
        #   - equivalently: if imef_data['count'][ir,ic] == 0:
        bool_idx = binnum == ibin
        if sum(bool_idx) == 0 or np.isnan(data[data_name][bool_idx].mean(dim='time').values[0])==True: # Theres an error here. I don't know why
            continue

        imef_data[data_name + '_mean'].loc[ir, ic] = data[data_name][bool_idx].mean(dim='time')
        imef_data[data_name + '_std'].loc[ir, ic] = data[data_name][bool_idx].std(dim='time')

    if created_file == True:
        # If the file already exists, average the new data with the existing data
        imef_data = average_data(imef_data, data_name, filename)

    return imef_data


def average_data(imef_data, data_name, filename):
    # Open file
    file_data = xr.open_dataset(filename)

    # Calculate weighted mean of incoming and existing data
    average_mean = (imef_data[data_name + '_count'] * imef_data[data_name + '_mean'] + file_data[data_name + '_count'] * file_data[data_name + '_mean']) / (
                           imef_data[data_name + '_count'] + file_data[data_name + '_count'])

    # For any bins that have 0 data points, the above divides by 0 and returns NaN. Change the NaN's to 0's
    average_mean = average_mean.fillna(0)

    # Calculate weighted standard deviation of incoming and existing data
    average_std = np.sqrt((imef_data[data_name + '_count'] * (
                imef_data[data_name + '_std'] ** 2 + (imef_data[data_name + '_mean'] - average_mean) ** 2) +
                           file_data[data_name + '_count'] * (file_data[data_name + '_std'] ** 2 + (
                        file_data[data_name + '_mean'] - average_mean) ** 2)) / \
                          (imef_data[data_name + '_count'] + file_data[data_name + '_count']))

    average_std = average_std.fillna(0)

    # Place the newly found averages to imef_data.
    # This could also just be made into a new Dataset, but it's easier this way.
    imef_data[data_name + '_count'].values = file_data[data_name + '_count'].values + imef_data[data_name + '_count'].values
    imef_data[data_name + '_mean'].values = average_mean.values
    imef_data[data_name + '_std'].values = average_std.values

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

    parser.add_argument('extra_data', type=str,
                        help='Data other than electric field data that the user wants downloaded and binned. Formatting: ex1,ex2,.... '
                             'If no extra data points, put None. Options for extra data are: Kp, IEF. More may be added later')

    parser.add_argument('driving_parameter', type=str,
                        help='Choose a driving parameter to separate the data by. Formatting: driving_parameter1:number_of_bins1'
                             'ex: [Kp,3]. For no driving parameter, put None. Options for driving parameters are: Kp, IEF. More may be added later')

    parser.add_argument('start_date', type=str, help='Start date of the data interval: ' '"YYYY-MM-DDTHH:MM:SS""')

    parser.add_argument('end_date', type=str, help='End date of the data interval: ''"YYYY-MM-DDTHH:MM:SS""')

    parser.add_argument('filename', type=str, help='Output file name')

    parser.add_argument('-n', '--no-show', help='Do not show the plot.', action='store_true')

    parser.add_argument('-e', '--exists',
                        help='The output file already exists, merge the data across the new dates with the existing data',
                        action='store_true')

    # If the polar plot is updated to spherical, update this note (and maybe change -p to -s)
    parser.add_argument('-p', '--polar', help='Convert the electric field values to polar', action='store_true')

    args = parser.parse_args()

    # Set up variables
    sc = args.sc
    mode = args.mode
    level = args.level

    # Start and end dates for download
    t0 = dt.datetime.strptime(args.start_date, '%Y-%m-%dT%H:%M:%S')
    t1 = dt.datetime.strptime(args.end_date, '%Y-%m-%dT%H:%M:%S')

    # Set up extra data and driving parameter arguments
    if args.extra_data == 'None' or args.extra_data == 'none':
        extra_data = [None]
    else:
        extra_data = args.extra_data.split(",")
        if type(extra_data) == str:
            extra_data = [extra_data]

    if args.driving_parameter == 'None' or args.driving_parameter == 'none':
        driving_parameter = [None]
    else:
        driving_parameter = args.driving_parameter.split(":")

    if driving_parameter[0] in extra_data and driving_parameter[0]!=None:
        raise AttributeError("The same variable cannot be binned and used as a driving parameter at the same time")

    # The name of the file where the data will be stored
    filename = args.filename
    # Whether the user wants the data in polar coordinates or not
    polar = args.polar

    # A datetime number to increment the dates by one day, but without extending into the next day
    one_day = dt.timedelta(days=1) - dt.timedelta(microseconds=1)

    # Ranges for L and MLT values to be binned for
    L_range = (4, 10)  # RE
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

    # Container which holds all of the above values. Cuts down of number of arguments passed in elsewhere
    L_and_MLT = LandMLT(L_range, MLT_range, dL, dMLT, L, MLT, nL, nMLT)

    # Boolean containing whether the file has been created
    if args.exists:
        created_file = True
    else:
        created_file = False

    # Download and process the data separately for each individual day,
    # So loop through every value in day
    while t0 < t1:
        # Assign the start time
        ti = t0

        # Determine the time difference between ti and midnight. Only will be a non-1 microsecond number on the first run through the loop
        # (Though it will not always be different on the first run)
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
            # The try/catch is a workaround for an error appearing when binning with a driving parameter on certain days.
            # Kp & 9/16/16 is an example. I don't know why it happens so I have to do this
            imef_data, created_file = prep_and_store_data(edi_data, fgm_data, mec_data, filename, polar, created_file,
                                                          L_and_MLT, ti, te, extra_data, driving_parameter)
            # try:
            #     imef_data, created_file = prep_and_store_data(edi_data, fgm_data, mec_data, filename, polar, created_file,
            #                                               L_and_MLT, ti, te, extra_data, driving_parameter)
            # except IndexError as indexexception:
            #     print('That weird error came up, I think:', indexexception)
            # except Exception as exception:
            #     raise exception

        # Increment the start day by an entire day, so that the next run in the loop starts on the next day
        t0 = ti + dt.timedelta(days=1) - timediff

    # Plot the data, unless specified otherwise
    if not args.no_show:
        # If the user chose to store edi data in polar coordinates, plot that data. Otherwise plot in cartesian
        if polar:
            xrplot.plot_efield_polar(nL, nMLT, imef_data)
        else:
            xrplot.plot_efield_cartesian(nL, nMLT, imef_data)


if __name__ == '__main__':
    main()
