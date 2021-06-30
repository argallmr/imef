import datetime as dt
import numpy as np
import cdflib
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
t0 = dt.datetime(2016, 8, 20, 0)
t1 = dt.datetime(2016, 8, 30, 0)
#t1 = t0 + dt.timedelta(hours=23.99)

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

mlt_vname = '_'.join((sc, 'mec', 'mlat'))
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

# Combine all into one data set
data = xr.merge([mec_data, edi_data])

# Format Data

df = pd.DataFrame(data=data['E_GSE'][:,0].values, columns=["EDI_X"])

# df['EDI_X'] = data['E_corrected'][:,0].values
df['EDI_Y'] = data['E_GSE'][:,1].values
df['EDI_Z'] = data['E_GSE'][:,2].values
df['MLT'] = data['MLT_sc'].values
df['L'] = data['L_sc'].values

df.to_pickle('data.pkl')

print(df)

print('complete')