import pandas as pd
import xarray as xr
from pymms.data import edi, util
from pymms.sdc import mrmms_sdc_api as api
from modeling.graphs import data_manipulation as dm
import numpy as np


def download_data_3d(sc, start_date,
                     end_date):  # For now srvy and level stay the same. probably make the fancy description better at some point if I need to.
    # Clean up imports when done here
    '''
    Parameters:
        sc: (str,list): Spacecraft IDs ('mms1', 'mms2', 'mms3', 'mms4')
        start_date (datetime): Start date of data interval, formatted as either %Y-%m-%d or
                          %Y-%m-%dT%H:%M:%S.
        end_date (datetime): End date of data interval, formatted as either %Y-%m-%d or
                        %Y-%m-%dT%H:%M:%S.
    Returns:
        df: Dataframe containing EDI and MEC data (MLT is azimuth, MLAT is altitude, L is radius)
    '''

    # Data parameters, these may be expanded and added to parameters
    mode = 'srvy'
    level = 'l2'

    # EDI Data
    edi_data = edi.load_data('mms1', 'srvy', optdesc='efield', start_date=start_date, end_date=end_date)
    edi_data = edi_data.rename({'Epoch': 'time'})

    # FGM Data (Not actually used?)
    # fgm_data = fgm.load_data('mms1', 'fgm', 'srvy', 'l2', t0, t1)

    # MEC Data
    sdc = api.MrMMS_SDC_API(sc, 'edp', 'fast', level, optdesc='dce', start_date=start_date, end_date=end_date)

    r_gse_vname = '_'.join((sc, 'mec', 'r', 'gse'))
    r_lbl_vname = '_'.join((sc, 'mec', 'r', 'gse', 'label'))

    v_gse_vname = '_'.join((sc, 'mec', 'v', 'gse'))
    v_lbl_vname = '_'.join((sc, 'mec', 'v', 'gse', 'label'))

    mlat_vname = '_'.join((sc, 'mec', 'mlat'))
    mlt_vname = '_'.join((sc, 'mec', 'mlt'))
    l_dip_vname = '_'.join((sc, 'mec', 'l', 'dipole'))

    sdc.instr = 'mec'
    sdc.mode = mode
    sdc.optdesc = 'epht89d'
    files = sdc.download()

    mec_data = util.cdf_to_ds(files[0], [r_gse_vname, v_gse_vname, mlt_vname, mlat_vname, l_dip_vname])
    mec_data = mec_data.rename({r_gse_vname: 'R_sc',  # convert R to polar
                                r_lbl_vname: 'R_sc_index',
                                v_gse_vname: 'V_sc',
                                v_lbl_vname: 'V_sc_index',
                                mlt_vname: 'MLT_sc',
                                mlat_vname: 'MLAT_sc',
                                l_dip_vname: 'L_sc',
                                'Epoch': 'time',
                                })

    # Interpolate to EDP times
    #   - MEC velocity is smoothly and slowly varying so can be up sampled
    #   - FGM data is sampled the same or faster so can be sychronized/down sampled
    mec_data = mec_data.interp_like(edi_data)

    # Combine all into one data set
    data = xr.merge([mec_data, edi_data])

    # Format Data

    df = pd.DataFrame(data=data['E_GSE'][:, 0].values, columns=["EDI_X"])

    # df['EDI_X'] = data['E_corrected'][:,0].values
    df['EDI_Y'] = data['E_GSE'][:, 1].values
    df['EDI_Z'] = data['E_GSE'][:, 2].values
    df['MLT'] = data['MLT_sc'].values
    df['MLAT'] = data['MLAT_sc'].values
    df['L'] = data['L_sc'].values

    return df


def download_data_2d(sc, start_date,
                     end_date):  # For now srvy and level stay the same. probably make the fancy description better at some point if I need to.
    # Clean up imports when done here
    '''
    Parameters:
        sc: (str,list): Spacecraft IDs ('mms1', 'mms2', 'mms3', 'mms4')
        start_date (datetime): Start date of data interval, formatted as either %Y-%m-%d or
                          %Y-%m-%dT%H:%M:%S.
        end_date (datetime): End date of data interval, formatted as either %Y-%m-%d or
                        %Y-%m-%dT%H:%M:%S.
    Returns:
        df: Dataframe containing EDI and MEC data (MLT is azimuth, L is radius)
    '''

    # Data parameters, these may be expanded and added to parameters
    mode = 'srvy'
    level = 'l2'

    # EDI Data
    edi_data = edi.load_data('mms1', 'srvy', optdesc='efield', start_date=start_date, end_date=end_date)
    edi_data = edi_data.rename({'Epoch': 'time'})

    # FGM Data (Not actually used?)
    # fgm_data = fgm.load_data('mms1', 'fgm', 'srvy', 'l2', t0, t1)

    # MEC Data
    sdc = api.MrMMS_SDC_API(sc, 'edp', 'fast', level, optdesc='dce', start_date=start_date, end_date=end_date)

    r_gse_vname = '_'.join((sc, 'mec', 'r', 'gse'))
    r_lbl_vname = '_'.join((sc, 'mec', 'r', 'gse', 'label'))

    v_gse_vname = '_'.join((sc, 'mec', 'v', 'gse'))
    v_lbl_vname = '_'.join((sc, 'mec', 'v', 'gse', 'label'))

    mlt_vname = '_'.join((sc, 'mec', 'mlt'))
    l_dip_vname = '_'.join((sc, 'mec', 'l', 'dipole'))

    sdc.instr = 'mec'
    sdc.mode = mode
    sdc.optdesc = 'epht89d'
    files = sdc.download()

    mec_data = util.cdf_to_ds(files[0], [r_gse_vname, v_gse_vname, mlt_vname, l_dip_vname])
    mec_data = mec_data.rename({r_gse_vname: 'R_sc',  # convert R to polar
                                r_lbl_vname: 'R_sc_index',
                                v_gse_vname: 'V_sc',
                                v_lbl_vname: 'V_sc_index',
                                mlt_vname: 'MLT_sc',
                                l_dip_vname: 'L_sc',
                                'Epoch': 'time',
                                })


    # Interpolate to EDP times
    #   - MEC velocity is smoothly and slowly varying so can be up sampled
    #   - FGM data is sampled the same or faster so can be sychronized/down sampled
    mec_data = mec_data.interp_like(edi_data)
    print(mec_data)
    print('oh boy')
    print(mec_data["MLT_sc"].values)

    # Combine all into one data set
    data = xr.merge([mec_data, edi_data])

    # Select required data (May remove at a later time, depending on what is needed)

    #selected_data = xr.merge([data['E_GSE'].to_dataset(), data['MLT_sc'].to_dataset(name="MLT"), data['L_sc'].to_dataset(name="L")])
    #selected_data.attrs = data.attrs

    # Convert edi_data to polar

    mec_data['r_polar'] = dm.cart2polar(mec_data['R_sc'])
    data['E_polar'] = (dm.rot2polar(data['E_GSE'], mec_data['r_polar'], 'E_index')
                           .assign_coords({'polar': ['r', 'phi']})
                           )

    #Not in polar coordinates
    df = pd.DataFrame(data=data['E_GSE'][:, 0].values, columns=["EDI_X"])

    # df['EDI_X'] = data['E_corrected'][:,0].values
    df['EDI_Y'] = data['E_GSE'][:, 1].values
    df['EDI_Z'] = data['E_GSE'][:, 2].values
    df['MLT'] = data['MLT_sc'].values
    df['L'] = data['L_sc'].values

    return df

