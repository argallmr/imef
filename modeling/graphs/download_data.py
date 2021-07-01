from pymms.data import edi, util, fgm, fpi
import numpy as np
import xarray as xr
from heliopy.data import omni

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


def get_edp_data(sc, level, ti, te):
    # Create the names of the variables that we will download
    # Time
    t_vname_fast = '_'.join((sc, 'edp', 'epoch', 'fast', level))
    t_vname_slow = '_'.join((sc, 'edp', 'epoch', 'slow', level))

    # fast efield data, and the label for the E_index
    e_fast_vname = '_'.join((sc, 'edp', 'dce', 'gse', 'fast', level))
    e_fast_vname_label = '_'.join((sc, 'edp', 'label1', 'fast', level))

    # slow efield data, and the label for the E_index
    e_slow_vname = '_'.join((sc, 'edp', 'dce', 'gse', 'slow', level))
    e_slow_vname_label = '_'.join((sc, 'edp', 'label1', 'slow', level))

    # Load edp_fast data
    edp_data_fast = util.load_data(sc, 'edp', 'fast', level, optdesc='dce', start_date=ti, end_date=te,
                              variables=[t_vname_fast, e_fast_vname, e_fast_vname_label])

    # Rename the variables
    edp_data_fast = edp_data_fast.rename({t_vname_fast: 'time',
                                          e_fast_vname: 'E',
                                          e_fast_vname_label: 'E_index'})

    # Load edp_slow data
    edp_data_slow = util.load_data(sc, 'edp', 'slow', level, optdesc='dce', start_date=ti, end_date=te,
                              variables=[t_vname_slow, e_slow_vname, e_slow_vname_label])

    # Rename the variables
    edp_data_slow = edp_data_slow.rename({t_vname_slow: 'time',
                                          e_slow_vname: 'E',
                                          e_slow_vname_label: 'E_index'})

    # Combine fast and slow data
    edp_data = xr.concat([edp_data_fast, edp_data_slow], dim='time')

    # Reorganize the data to sort by time
    edp_data = edp_data.sortby('time')

    return edp_data


def get_omni_data(ti, te):
    # Download the omni data as a timeseries object
    full_omni_data = omni.hro2_1min(ti, te)

    # Convert time series object to an xarray dataset
    full_omni_data = full_omni_data.to_dataframe().to_xarray()

    # We only want the V and B values in this xarray. We also want to create new indices where each individual component of V and B will reside
    # Ex: V_index = (Vx, Vy, Vz)
    omni_data = xr.Dataset(coords={'Time': full_omni_data['Time'], 'V_index': ['Vx', 'Vy', 'Vz'], 'B_index': ['Bx', 'By', 'Bz']})

    # Format the V and B values so that they will fit inside the above dataset.
    V_values = np.vstack((full_omni_data['Vx'], full_omni_data['Vy'], full_omni_data['Vz'])).T
    B_values = np.vstack((full_omni_data['BX_GSE'], full_omni_data['BY_GSE'], full_omni_data['BZ_GSE'])).T

    # Put the values into the new dataset
    omni_data['V'] = xr.DataArray(V_values, dims=['Time', 'V_index'], coords={'Time': omni_data['Time']})
    omni_data['B'] = xr.DataArray(B_values, dims=['Time', 'B_index'], coords={'Time': omni_data['Time']})

    # Rename time so that it is the same as the other datasets, making concatenation easier
    omni_data = omni_data.rename({'Time': 'time'})

    return omni_data


def get_dis_data(sc, mode, level, ti, te):
    # Download dis_data
    full_dis_data = fpi.load_moms(sc, mode, level, 'dis-moms', ti, te)

    # Slice the velocity data
    V_data = full_dis_data['velocity']

    # Make a new dataset for the velocity data to go into, with renamed variables
    dis_data = xr.Dataset(coords={'time': full_dis_data['time'], 'V_index': ['Vx', 'Vy', 'Vz']})

    # Place the velocity data into the new dataset
    dis_data['V'] = xr.DataArray(V_data, dims=['time', 'V_index'], coords={'time': full_dis_data['time']})

    return dis_data


def get_des_data(sc, mode, level, ti, te):
    # Download des_data
    full_des_data = fpi.load_moms(sc, mode, level, 'des-moms', ti, te)

    # Slice the velocity data
    V_data = full_des_data['velocity']

    # Make a new dataset for the velocity data to go into, with renamed variables
    des_data = xr.Dataset(coords={'time': full_des_data['time'], 'V_index': ['Vx', 'Vy', 'Vz']})

    # Place the velocity data into the new dataset
    des_data['V'] = xr.DataArray(V_data, dims=['time', 'V_index'], coords={'time': full_des_data['time']})

    return des_data
