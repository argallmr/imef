import numpy as np
import xarray as xr
import datetime as dt
from scipy.stats import binned_statistic

# For debugging purposes
np.set_printoptions(threshold=np.inf)


def cart2polar(pos_cart, factor=1):
    '''
    Rotate cartesian position coordinates to polar coordinates
    '''
    r = (np.sqrt(np.sum(pos_cart[:, [0, 1]] ** 2, axis=1)) * factor)
    phi = (np.arctan2(pos_cart[:, 1], pos_cart[:, 0]))
    pos_polar = xr.concat([r, phi], dim='polar').T.assign_coords({'polar': ['r', 'phi']})

    return pos_polar


def rot2polar(vec, pos, dim):
    '''
    Rotate vector from cartesian coordinates to polar coordinates
    '''
    # Polar unit vectors
    phi = pos.loc[:, 'phi']
    r_hat = xr.concat([np.cos(phi).expand_dims(dim), np.sin(phi).expand_dims(dim)], dim=dim)
    phi_hat = xr.concat([-np.sin(phi).expand_dims(dim), np.cos(phi).expand_dims(dim)], dim=dim)

    # Rotate vector to polar coordinates
    Vr = vec[:, [0, 1]].dot(r_hat, dims=dim)
    Vphi = vec[:, [0, 1]].dot(phi_hat, dims=dim)
    v_polar = xr.concat([Vr, Vphi], dim='polar').T.assign_coords({'polar': ['r', 'phi']})

    return v_polar


def remove_spacecraft_efield(edi_data, fgm_data, mec_data):
    # E = v x B, 1e-3 converts units to mV/m
    E_sc = 1e-3 * np.cross(mec_data['V_sc'][:, :3], fgm_data['B_GSE'][:, :3])

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


def slice_data_by_time(full_data, ti, te):
    # Here is where the desired time and data values will be placed
    time = np.array([])
    wanted_value = np.array([])

    # Slice the wanted data and put them into 2 lists
    for counter in range(0, len(full_data)):
        # The data at each index is all in one line, separated by whitespace. Separate them
        new = str.split(full_data.iloc[counter][0])

        # Create the time at that point
        time_str = str(new[0]) + '-' + str(new[1]) + '-' + str(new[2]) + 'T' + str(new[3][:2]) + ':00:00'

        # Make a datetime object out of time_str
        insert_time_beg = dt.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')

        # We know for Kp that the middle of the bin is 1.5 hours past the beginning of the bin (which is insert_time_beg)
        insert_time_mid = insert_time_beg + dt.timedelta(hours=1.5)

        # If the data point is within the time range that is desired, insert the time and the associated kp index
        # The other datasets use the dates as datetime64 objects. So must use those instead of regular datetime objects
        if insert_time_mid + dt.timedelta(hours=1.5) > ti and insert_time_mid - dt.timedelta(hours=1.5) < te:
            insert_kp = new[7]
            time = np.append(time, [insert_time_mid])
            wanted_value = np.append(wanted_value, [insert_kp])

    return time, wanted_value


def expand_5min(time, kp):
    # The assumption here is that they are 3 hour bins, since this is made for Kp. Could be generalized later

    # Expanding the kp values is easy, as np.repeat does this for us.
    # Eg: np.repeat([1,2,3],3) = [1,1,1,2,2,2,3,3,3]
    new_kp = np.repeat(kp, 36)

    new_times = np.array([])

    # Iterate through every time that is given
    for a_time in time:
        # Put the first time value into the xarray. This corresponds to the start of the kp window
        new_times = np.append(new_times, [a_time - dt.timedelta(hours=1.5)])

        # There are 36 5-minute intervals in the 3 hour window Kp covers. We have the time in the middle of the 3 hour window.
        # Here all the values are created (other than the start of the window, done above) by incrementally getting 5 minutes apart on both sides of the middle value. This is done 18 times
        # However we must make an exception when the counter is 0, because otherwise it will put the middle of the window twice
        for counter in range(18):
            if counter == 0:
                new_time = a_time
                new_times = np.append(new_times, [new_time])
            else:
                new_time_plus = a_time + counter * dt.timedelta(minutes=5)
                new_time_minus = a_time - counter * dt.timedelta(minutes=5)
                new_times = np.append(new_times, [new_time_plus, new_time_minus])

    # The datetime objects we want are created, but out of order. Put them in order
    new_times = np.sort(new_times, axis=0)

    return new_times, new_kp


def interpolate_data_like(data, data_like):
    data = data.interp_like(data_like)

    return data


def create_timestamps(data, vars_to_bin, ti, te):
    # Define the epoch and one second in np.datetime 64. This is so we can convert np.datetime64 objects to timestamp values
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')

    # Round up the end time by one microsecond so the bins aren't marginally incorrect
    te = te + dt.timedelta(microseconds=1)

    # Convert the start and end times to a unix timestamp
    # This section adapts for the 4 hour time difference (in seconds) that timestamp() automatically applies to datetime. Otherwise the times from data and these time will not line up right
    # This appears because timestamp() corrects for local time difference, while the np.datetime64 method did not
    # This could be reversed and added to all the times in data, but I chose this way.
    ti = ti.timestamp() - 14400
    te = te.timestamp() - 14400

    # Create the array where the unix timestamp values go
    # The timestamp values are needed so we can bin the values with binned_statistic
    timestamps = (data['time'].values - unix_epoch) / one_second

    # Get the times here. This way we don't have to rerun getting the times for every single variable that is being binned
    count, bin_edges, binnum = binned_statistic(x=timestamps, values=data[vars_to_bin[0]], statistic='count', bins=288,
                                               range=(ti, te))

    # Create an nparray where the new 5 minute interval datetime64 objects will go
    new_times = np.array([], dtype=object)

    # Create the datetime64 objects and add them to new_times
    for time in bin_edges:
        # Don't run if the time value is the last index in bin_edges. There is 1 more bin edge than there is mean values
        # This is because bin_edges includes an extra edge to encompass all the means
        # As a result, the last bin edge (when shifted to be in the middle of the dataset) doesn't correspond to a mean value
        # So it must be ignored so that the data will fit into a new dataset
        if time != bin_edges[-1]:
            # Convert timestamp to datetime object
            new_time = dt.datetime.utcfromtimestamp(time)

            # Add 2.5 minutes to place the time in the middle of each bin, rather than the beginning
            new_time = new_time + dt.timedelta(minutes=2.5)

            # Add the object to the nparray
            new_times = np.append(new_times, [new_time])

    # Return timestamp versions of ti, te, and the datetime64 objects.
    # Also return the datetime64 objects of the 5 minute intervals created in binned_statistic
    return ti, te, timestamps, new_times


def bin_5min(data, vars_to_bin, index_names, ti, te):
    # The assumption with this function is that exactly 1 day of data is being inputted. Otherwise this will not work properly, as the number of bins will be incorrect
    # There is probably a simple fix to this, but it isn't implemented
    # Also, any variables that are not in var_to_bin are lost (As they can't be mapped to the new times otherwise)

    # In order to bin the values properly, we need to convert the datetime objects to integers. I chose to use unix timestamps to do so
    ti, te, timestamps, new_times = create_timestamps(data, vars_to_bin, ti, te)

    # Iterate through every variable (and associated index) in the given list
    for var_counter in range(len(vars_to_bin)):
        if index_names[var_counter] == '':
            # Since there is no index associated with this variable, there is only 1 thing to be meaned. So take the mean of the desired variable
            means, bin_edges_again, binnum = binned_statistic(x=timestamps, values=data[vars_to_bin[var_counter]], statistic='mean', bins=288, range=(ti, te))
            std, bin_edges_again, binnum = binned_statistic(x=timestamps, values=data[vars_to_bin[var_counter]], statistic='std', bins=288, range=(ti, te))

            # Create the dataset for the meaned variable
            new_data = xr.Dataset(coords={'time': new_times})

            # Fix the array so it will fit into the dataset
            var_values = means.T
            var_values_std = std.T

            # Put the data into the dataset
            new_data[vars_to_bin[var_counter]] = xr.DataArray(var_values, dims=['time'], coords={'time': new_times})
            new_data[vars_to_bin[var_counter]+'_std'] = xr.DataArray(var_values_std, dims=['time'], coords={'time': new_times})
        else:
            # Empty array where the mean of the desired variable will go
            means = np.array([[]])
            stds = np.array([[]])

            # Iterate through every variable in the associated index
            for counter in range(len(data[index_names[var_counter] + '_index'])):
                # Find the mean of var_to_bin
                # mean is the mean in each bin, bin_edges is the edges of each bin in timestamp values, and binnum is which values go in which bin
                mean, bin_edges_again, binnum = binned_statistic(x=timestamps, values=data[vars_to_bin[var_counter]][:, counter], statistic='mean', bins=288, range=(ti, te))
                std, bin_edges_again, binnum = binned_statistic(x=timestamps, values=data[vars_to_bin[var_counter]][:, counter], statistic='std', bins=288, range=(ti, te))

                # If there are no means yet, designate the solved mean value as the array where all of the means will be stored. Otherwise combine with existing data
                if means[0].size == 0:
                    means = [mean]
                    stds = [std]
                else:
                    means = np.append(means, [mean], axis=0)
                    stds = np.append(stds, [std], axis=0)

            # Create the new dataset where the 5 minute bins will go
            new_data = xr.Dataset(
                coords={'time': new_times, index_names[var_counter] + '_index': data[index_names[var_counter] + '_index']})

            # Format the mean values together so that they will fit into new_data
            var_values = means.T
            var_values_std = stds.T

            # Put in var_values
            new_data[vars_to_bin[var_counter]] = xr.DataArray(var_values, dims=['time', index_names[var_counter] + '_index'], coords={'time': new_times})
            new_data[vars_to_bin[var_counter]+'_std'] = xr.DataArray(var_values_std, dims=['time', index_names[var_counter] + '_index'], coords={'time': new_times})

        # If this is the first run, designate the created data as the dataset that will hold all the data. Otherwise combine with the existing data
        if var_counter == 0:
            complete_data = new_data
        else:
            complete_data = xr.merge([complete_data, new_data])

    return complete_data
