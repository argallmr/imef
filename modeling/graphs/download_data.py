from pymms.data import edi, util, fgm
from pymms.sdc import mrmms_sdc_api as api
import numpy as np
import xarray as xr

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
    t_vname_fast = '_'.join((sc, 'edp', 'epoch', 'fast', level))
    t_vname_slow = '_'.join((sc, 'edp', 'epoch', 'slow', level))
    e_fast_vname = '_'.join((sc, 'edp', 'dce', 'gse', 'fast', level))
    e_fast_vname_label = '_'.join((sc, 'edp', 'label1', 'fast', level))
    e_slow_vname = '_'.join((sc, 'edp', 'dce', 'gse', 'slow', level))
    e_slow_vname_label = '_'.join((sc, 'edp', 'label1', 'slow', level))
    edp_data_fast = util.load_data(sc, 'edp', 'fast', level, optdesc='dce', start_date=ti, end_date=te,
                              variables=[t_vname_fast, e_fast_vname, e_fast_vname_label])

    edp_data_fast = edp_data_fast.rename({t_vname_fast: 'time',
                                          e_fast_vname: 'E',
                                          e_fast_vname_label: 'E_index'})

    edp_data_slow = util.load_data(sc, 'edp', 'slow', level, optdesc='dce', start_date=ti, end_date=te,
                              variables=[t_vname_slow, e_slow_vname, e_slow_vname_label])

    edp_data_slow = edp_data_slow.rename({t_vname_slow: 'time',
                                          e_slow_vname: 'E',
                                          e_slow_vname_label: 'E_index'})

    edp_data = xr.concat([edp_data_fast, edp_data_slow], dim='time')
    edp_data = edp_data.sortby('time')

    return edp_data
