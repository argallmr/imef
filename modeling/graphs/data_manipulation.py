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


def slice_data_by_time(full_data, ti, te):
    # Here is where the desired time and data values will be placed
    time = np.array([])
    wanted_value = np.array([])

    # Slice the wanted data and put them into 2 lists
    for counter in range(0, len(full_data)):
        # The data at each index is all in one line, separated by whitespace. Separate them
        new = str.split(full_data.iloc[counter][0])

        # Create the time at that point. This could probably be streamlined
        time_str = str(new[0]) + '-' + str(new[1]) + '-' + str(new[2]) + 'T' + str(new[3][:2]) + ':00:00'

        # Make a datetime object out of time_str
        insert_time_beg = dt.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')

        # We know for Kp that the middle of the bin is 1.5 hours past the beginning of the bin (which is insert_time_beg)
        insert_time_mid = insert_time_beg + dt.timedelta(hours=1.5)

        # If the data point is within the time range that is desired, insert the time and the associated kp index
        if insert_time_mid + dt.timedelta(hours=1.5) >= ti and insert_time_mid - dt.timedelta(hours=1.5) <= te:
            insert_kp = new[7]
            time = np.append(time, [insert_time_mid])
            wanted_value = np.append(wanted_value, [insert_kp])

    return time, wanted_value


def bin_5min(data, ti, te):
    # The assumption with this function is that exactly 1 day of data is being inputted. Otherwise this will not work properly, as the number of bins will be incorrect
    # There is probably a simple fix to this, but it isn't implemented

    # Create the array where the unix timestamp values will go.
    # The timestamp values are needed so we can bin the values with binned_statistic
    timestamp_values = np.array([])

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

    # Convert all the time values from
    for the_time in data['time'].values:
        timestamp = (the_time - unix_epoch) / one_second
        timestamp_values = np.append(timestamp_values, [timestamp])

    # Find the mean of Vx, Vy, Vz
    # mean is the mean in each bin, bin_edges is the edges of each bin in timestamp values, and binnum is which values go in which bin
    mean_x, bin_edges_x, binnum_x = binned_statistic(x=timestamp_values, values=data['V'][:, 0], statistic='mean', bins=288, range=(ti, te))
    mean_y, bin_edges_y, binnum_y = binned_statistic(x=timestamp_values, values=data['V'][:, 1], statistic='mean', bins=288, range=(ti, te))
    mean_z, bin_edges_z, binnum_z = binned_statistic(x=timestamp_values, values=data['V'][:, 2], statistic='mean', bins=288, range=(ti, te))

    # Create an nparray where the new 5 minute interval datetime64 objects will go
    new_times = np.array([], dtype=object)

    # Create the datetime64 objects and add them to new_times
    for time in bin_edges_x:
        # Don't run if the time value is the last index in bin_edges. There is 1 more bin edge than there is mean values
        # This is because bin_edges includes an extra edge to encompass all the means
        # As a result, the last bin edge (when shifted to be in the middle of the dataset) doesn't correspond to a mean value
        # So it must be removed so that the data will fit into a new dataset
        if time != bin_edges_x[-1]:
            # Convert timestamp to datetime object
            new_time = dt.datetime.utcfromtimestamp(time)

            # Add 2.5 minutes to place the time in the middle of each bin, rather than the beginning
            new_time = new_time + dt.timedelta(minutes=2.5)

            # Convert the datetime object to a datetime64 object
            new_time = np.datetime64(new_time)

            # Add the object to the nparray
            new_times = np.append(new_times, [new_time])

    # Create the new dataset where the 5 minute bins will go
    new_data = xr.Dataset(coords={'time': new_times, 'V_index': ['Vx', 'Vy', 'Vz']})

    # Format the mean values together so that they will fit into new_data
    V_values = np.vstack((mean_x, mean_y, mean_z)).T

    # Put in V_values
    new_data['V'] = xr.DataArray(V_values, dims=['time', 'V_index'], coords={'time': new_times})

    return new_data
