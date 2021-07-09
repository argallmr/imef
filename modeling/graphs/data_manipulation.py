import numpy as np
import xarray as xr
import datetime as dt
from scipy.stats import binned_statistic


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


def bin_5min(data):
    print('Put this in boyo')
    #statistics = binned_statistic(x=data['time'], values=data['V'])
