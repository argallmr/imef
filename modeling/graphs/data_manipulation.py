import numpy as np
import xarray as xr


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
