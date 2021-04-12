import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import pdb
from pymms.data import edi, edp, fgm, util
from pymms.sdc import mrmms_sdc_api as api

# Data parameters
sc = 'mms1'
mode = 'srvy'
level = 'l2'

# Time Parameters
t0 = dt.datetime(2020, 5, 30, 0)
#t1 = dt.datetime(2020, 6, 30, 0)
t1 = t0 + dt.timedelta(hours=23.99)

# EDI Data
edi_data = edi.load_data('mms1', 'srvy', optdesc='efield', start_date=t0, end_date=t1)
edi_data = edi_data.rename({'Epoch': 'time'})

# FGM Data
fgm_data = fgm.load_data('mms1', 'fgm', 'srvy', 'l2', t0, t1)

# MEC Data
sdc = api.MrMMS_SDC_API(sc, 'edp', 'fast', level, optdesc='dce', start_date=t0, end_date=t1)

r_gse_vname = '_'.join((sc, 'mec', 'r', 'gse'))
r_lbl_vname = '_'.join((sc, 'mec', 'r', 'gse', 'label'))

v_gse_vname = '_'.join((sc, 'mec', 'v', 'gse'))
v_lbl_vname = '_'.join((sc, 'mec', 'v', 'gse', 'label'))

sdc.instr = 'mec'
sdc.mode = mode
sdc.optdesc = 'epht89d'
files = sdc.download()

mec_data = util.cdf_to_ds(files[0], [r_gse_vname, v_gse_vname])
mec_data = mec_data.rename({r_gse_vname: 'R_sc',
                            r_lbl_vname: 'R_sc_index',
                            v_gse_vname: 'V_sc',
                            v_lbl_vname: 'V_sc_index',
                            'Epoch': 'time'
                            })

mec_data = mec_data.assign_coords(R_sc_index=['Rx', 'Ry', 'Rz'],
                                  V_sc_index=['Vx', 'Vy', 'Vz'])

# EDP Fast Data
t_vname = '_'.join((sc, 'edp', 'epoch', 'fast', level))
e_fast_vname = '_'.join((sc, 'edp', 'dce', 'gse', 'fast', level))
e_labl_vname = '_'.join((sc, 'edp', 'label1', 'fast', level))
sdc = api.MrMMS_SDC_API(sc, 'edp', 'fast', level, optdesc='dce', start_date=t0, end_date=t1)
files = sdc.download()
edp_fast_data = util.cdf_to_ds(files[0], e_fast_vname)
edp_fast_data = edp_fast_data.rename({t_vname: 'time',
                                      e_fast_vname: 'E',
                                      e_labl_vname: 'E_index'})

# EDP Slow Data
t_vname = '_'.join((sc, 'edp', 'epoch', 'slow', level))
e_slow_vname = '_'.join((sc, 'edp', 'dce', 'gse', 'slow', level))
e_labl_vname = '_'.join((sc, 'edp', 'label1', 'slow', level))
sdc.mode = 'slow'
files = sdc.download()
edp_slow_data = util.cdf_to_ds(files[0], e_slow_vname)
edp_slow_data = edp_slow_data.rename({t_vname: 'time',
                                      e_slow_vname: 'E',
                                      e_labl_vname: 'E_index'})

# Combine slow and fast data
edp_data = xr.concat([edp_fast_data, edp_slow_data], dim='time')
edp_data = edp_data.sortby('time')

# Interpolate to EDP times
#   - MEC velocity is smoothly and slowly varying so can be up sampled
#   - FGM data is sampled the same or faster so can be sychronized/down sampled
fgm_data = fgm_data.interp_like(edp_data)
mec_data = mec_data.interp_like(edp_data)
edi_data = edi_data.interp_like(edp_data)

# Compute the spacecraft electric field
#   - E = v x B, 1e-3 converts units to mV/m
E_sc = 1e-3 * np.cross(mec_data['V_sc'], fgm_data['B_GSE'][:,:3])
E_sc = xr.DataArray(E_sc,
                    dims=['time', 'E_index'],
                    coords={'time': edp_data['time'],
                            'E_index': ['Ex', 'Ey', 'Ez']},
                    name='E_sc')

# Remove E_sc from the measured electric field
E_corrected = edp_data['E'] - E_sc
E_corrected.name = 'E_corrected'

# Magnitude of the corrected electric field
E_corrected_mag = xr.DataArray(np.linalg.norm(E_corrected,
                                              axis=E_corrected.get_axis_num('E_index')),
                               dims='time',
                               coords={'time': edp_data['time']},
                               name='|E_corrected|')

# Combine all into one data set
data = xr.merge([fgm_data, edi_data, E_corrected])

# Format Data

df = pd.DataFrame(data=data['B_GSE'].values, columns=["B_X", "B_Y", "B_Z", "B_?"])

df['EDP_X'] = data['E_corrected'][:,0].values
df['EDP_Y'] = data['E_corrected'][:,1].values
df['EDP_Z'] = data['E_corrected'][:,2].values

df['EDI_X'] = data['E_GSE'][:,0].values
df['EDI_Y'] = data['E_GSE'][:,1].values
df['EDI_Z'] = data['E_GSE'][:,2].values
df['IsEdi'] = df['EDI_X'].apply(lambda x: 1 if np.abs(x) > 0 else 0)

df = df.dropna(subset=['EDP_X'])

df.to_pickle('train_data.pkl')

print(df)

print('Complete')