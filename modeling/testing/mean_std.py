# Calculate the mean and standard deviation of edp and fgm data over an interval

import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import pdb
from scipy import stats
from pymms.data import edi, edp, fgm, util
from pymms.sdc import mrmms_sdc_api as api

# Data parameters
sc = 'mms1'
mode = 'srvy'
level = 'l2'

# Time Parameters
t0 = dt.datetime(2020, 5, 30, 0)
t1 = t0 + dt.timedelta(hours=23.99)

# EDI Data
edi_data = edi.load_data('mms1', 'srvy', optdesc='efield', start_date=t0, end_date=t1)
edi_data = edi_data.rename({'Epoch': 'time'})

# Variable names
tm_vname = '_'.join((sc, 'edi', 't', 'delta', 'minus', mode, level))
e_vname = '_'.join((sc, 'edi', 'e', 'gse', mode, level))

# Timestamps begin on 0's and 5's and span 5 seconds. The timestamp is at
# the weighted mean of all beam hits. To get the beginning of the timestamp,
# subtract the time's DELTA_MINUS. But this is inaccurate by a few nanoseconds
# so we have to round to the nearest second.
edi_time = edi_data['time'] - edi_data[tm_vname].astype('timedelta64[ns]')
edi_time = [(t - tdelta)
            if tdelta.astype(int) < 5e8
            else (t + np.timedelta64(1, 's') - tdelta)
            for t, tdelta in zip(edi_time.data, edi_time.data - edi_time.data.astype('datetime64[s]'))
           ]
edi_data['time'] = edi_time

# FGM Data
fgm_data = fgm.load_data('mms1', 'fgm', 'srvy', 'l2', t0, t1)

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

#Interpolating data
fgm_data = fgm_data.interp_like(edi_data)
edp_data = edp_data.interp_like(edi_data)

print(edi_data['E_GSE'][:,0])

# Calculating the mean and standard deviation
fgm_means, bin_edges, bin_num = stats.binned_statistic(edi_data['E_GSE'][:,0], fgm_data['B_GSE'][:,0], statistic='mean', bins=10)
fgm_std, bin_edges, bin_num = stats.binned_statistic(edi_data['E_GSE'][:,0], fgm_data['B_GSE'][:,0], statistic='std', bins=10)

edp_means, bin_edges, bin_num = stats.binned_statistic(edi_data['E_GSE'][:,0], edp_data['E'][:,0], statistic='mean', bins=10)
edp_std, bin_edges, bin_num = stats.binned_statistic(edi_data['E_GSE'][:,0], edp_data['E'][:,0], statistic='std', bins=10)

# Printing results

#print('FGM DATA\nMean\n', fgm_means, '\nStandard Deviation\n', fgm_std)
#print('EDP DATA\nMean\n', edp_means, '\nStandard Deviation\n', edp_std)

