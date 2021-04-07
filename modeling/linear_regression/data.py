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

# FGM Data
fgm_data = fgm.load_data('mms1', 'fgm', 'srvy', 'l2', t0, t1)
fgm_data = fgm_data.rename({'time': 'Epoch'})

# EDP Fast Data
t_vname = '_'.join((sc, 'edp', 'epoch', 'fast', level))
e_fast_vname = '_'.join((sc, 'edp', 'dce', 'gse', 'fast', level))
e_labl_vname = '_'.join((sc, 'edp', 'label1', 'fast', level))
sdc = api.MrMMS_SDC_API(sc, 'edp', 'fast', level, optdesc='dce', start_date=t0, end_date=t1)
files = sdc.download()
edp_fast_data = util.cdf_to_ds(files[0], e_fast_vname)
edp_fast_data = edp_fast_data.rename({t_vname: 'Epoch',
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

"""
# Combine all into one data set
data = xr.merge([fgm_data, edi_data, E_corrected])

#data.to_netcdf(path="data1.nc") unknown error


# Format Data

df = pd.DataFrame(data=data['B'].values, columns=["B_X", "B_Y", "B_Z", "B_?"])

df['EDP_X'] = data['E_corrected'][:,0].values
df['EDP_Y'] = data['E_corrected'][:,1].values
df['EDP_Z'] = data['E_corrected'][:,2].values

df['EDI_X'] = data['E_GSE'][:,0].values
df['EDI_Y'] = data['E_GSE'][:,1].values
df['EDI_Z'] = data['E_GSE'][:,2].values
df['IsEdi'] = df['EDI_X'].apply(lambda x: 1 if np.abs(x) > 0 else 0)

#df.to_pickle('train_data.pkl')
df.to_csv('dummy_data.csv')
"""
