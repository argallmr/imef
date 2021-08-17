import download_data as dd
import datetime as dt
import argparse
import numpy as np
from data_manipulation import remove_spacecraft_efield, remove_corot_efield
import xarray as xr

# For debugging purposes
np.set_printoptions(threshold=np.inf)

# If numbers overflow, raise an error instead of raising a warning
np.seterr(over='raise')

# Note that it is possible that NaN values appear in the final xarray object. This is because there were no data points in those bins
# To remove these values for a single variable, use xarray_object = xarray_object.where(np.isnan(xarray_object['variable_name']) == False, drop=True) (Variable has no indices)
# Or xarray_object = xarray_object.where(np.isnan(xarray_object['variable_name'][:,0]) == False, drop=True) (With indices)
# Repeat this for each variable to remove all nan values from the final product

# To Do -> When next run of this is complete, check for overflow errors, determine which variables need to be cast to float64, and do so

def main():
    # Take arguments and set up variables
    parser = argparse.ArgumentParser(
        description='Download lots of data, and store the data into a netCDF (.nc) file'
    )

    parser.add_argument('sc', type=str, help='Spacecraft Identifier')

    parser.add_argument('mode', type=str, help='Data rate mode')

    parser.add_argument('level', type=str, help='Data level')

    parser.add_argument('start_date', type=str, help='Start date of the data interval: ' '"YYYY-MM-DDTHH:MM:SS""')

    parser.add_argument('end_date', type=str, help='End date of the data interval: ''"YYYY-MM-DDTHH:MM:SS""')

    parser.add_argument('filename', type=str, help='Output file name')

    parser.add_argument('-e', '--exists', help='The output file already exists, merge the data across the new dates with the existing data', action='store_true')

    args = parser.parse_args()

    sc = args.sc
    mode = args.mode
    level = args.level
    start = dt.datetime.strptime(args.start_date, '%Y-%m-%dT%H:%M:%S')
    end = dt.datetime.strptime(args.end_date, '%Y-%m-%dT%H:%M:%S')

    t0 = start

    # Name of the file where the sampled data will go
    filename = args.filename

    # Boolean containing whether the file has been created
    if args.exists:
        created_file = True
    else:
        created_file = False

    RE = 6371  # km. This is the conversion from km to Earth radii

    # A datetime number to increment the dates by one day, but without extending into the next day
    one_day = dt.timedelta(days=1) - dt.timedelta(microseconds=1)

    while t0 < end:
        # Assign the start time
        ti = t0

        # Determine the time difference between ti and midnight. Only will be a non-zero number on the first run through the loop
        timediff = dt.datetime.combine(dt.date.min, ti.time()) - dt.datetime.min

        # Assign the end date. timediff is used so that data is only downloaded from 1 day per run through the loop, which prevents bugs from appearing
        te = ti + one_day - timediff

        # If te were to extend past the desired end date, make te the desired end date
        if te > end:
            te = end

        print(ti, '%%', te)

        try:
            # Read EDI data
            edi_data = dd.get_edi_data(sc, mode, level, ti, te, binned=True)

            # Read FGM data
            fgm_data = dd.get_fgm_data(sc, mode, ti, te, binned=True)

            # Read MEC data
            mec_data = dd.get_mec_data(sc, mode, level, ti, te, binned=True)

            # Read EDP data
            edp_data = dd.get_edp_data(sc, level, ti, te, binned=True)

            # Read OMNI data
            # There is no standard deviation for omni, since it is an outside package that does the binning itself. I don't think there is a way to get it with a command from the package
            omni_data = dd.get_omni_data(ti, te)

            # Read DIS data
            dis_data = dd.get_dis_data(sc, mode, level, ti, te, binned=True)

            # Read DES data
            des_data = dd.get_des_data(sc, mode, level, ti, te, binned=True)

            # There are times where x, y, and z are copied, but the corresponding values are not, resulting in 6 coordinates
            # This throws an error when trying to computer the cross product in remove_spacecraft_efield, so raise an error here if this happens
            # Only seems to happen on one day, 12/18/16
            if len(mec_data["V_sc_index"]) != 3:
                raise ValueError("There should be 3 coordinates in V_sc_index")

        except Exception as ex:
            # Download will return an error when there is no data in the files that are being read.
            # Catch the error and print it out
            # Maybe other errors with the new data too? Who knows. Only thoroughly tested for first 3 downloads
            print('Failed: ', ex)
        else:
            # Omni data was sampled at 5 minute intervals, which is what we want. We map all the other datasets onto omni_data so that they are all on the same times
            # The spacecraft creates its own electric field and must be removed from the total calculations
            edi_data = remove_spacecraft_efield(edi_data, fgm_data, mec_data)

            # Remove the corotation electric field
            edi_data = remove_corot_efield(edi_data, mec_data, RE)

            # Combine all of the data into one dataset
            one_day_data = xr.merge([edi_data, fgm_data, mec_data, edp_data, omni_data, dis_data, des_data])

            # By binning the data, these coordinates become incorrect. They also don't seem to be needed, so just remove them
            one_day_data = one_day_data.drop_vars("Epoch_plus_var")
            one_day_data = one_day_data.drop_vars("Epoch_minus_var")

            # If there is no existing data, set this day's data as the existing data. Otherwise combine with existing data
            if created_file == False:
                complete_data = one_day_data
                complete_data.to_netcdf(filename)
                created_file = True
            else:
                complete_data = xr.concat([complete_data, one_day_data], 'time')
                complete_data.to_netcdf(filename)

        # Increment the start day by an entire day, so that the next run in the loop starts on the next day
        t0 = ti + dt.timedelta(days=1) - timediff

    # Download the kp data. This is done outside the loop since it downloads all the data for 1 year at once, so repeating per day would be very inefficient
    kp_data = dd.get_kp_data(start, end, complete_data['time'].values)

    # Merge the kp data with the rest of the data
    complete_data = xr.merge([complete_data, kp_data])

    # Output the data to a file with the name given by the user
    complete_data.to_netcdf(args.filename)

if __name__ == '__main__':
    main()
