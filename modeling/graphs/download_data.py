import datetime as dt
import numpy as np
import xarray as xr
import pdb
from pymms.data import edi, fgm, util
from pymms.sdc import mrmms_sdc_api as api

# Data parameters
sc = 'mms1'
mode = 'srvy'
level = 'l2'

# Time Parameters
t0 = dt.datetime(2020, 5, 30, 0)
t1 = dt.datetime(2020, 6, 30, 0)
#t1 = t0 + dt.timedelta(hours=23.99)

# EDI Data
edi_data = edi.load_data('mms1', 'srvy', optdesc='efield', start_date=t0, end_date=t1)

# FGM Data
fgm_data = fgm.load_data(sc, mode, t0, t1)

fgm_data = fgm_data.rename({'time': 'Epoch'})

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
                            })

mec_data = mec_data.assign_coords(R_sc_index=['Rx', 'Ry', 'Rz'],
                                  V_sc_index=['Vx', 'Vy', 'Vz'])

# Interpolate to EDP times
#   - MEC velocity is smoothly and slowly varying so can be up sampled
#   - FGM data is sampled the same or faster so can be sychronized/down sampled
fgm_data = fgm_data.interp_like(edi_data)
mec_data = mec_data.interp_like(edi_data)

#pdb.set_trace()

E_sc = 1e-3 * np.cross(mec_data['V_sc'], fgm_data['B'][:,:3])

E_sc = xr.DataArray(E_sc,
                    dims=['Epoch', 'E_index'],
                    coords={'Epoch': edi_data['Epoch'],
                            'E_index': ['Ex', 'Ey', 'Ez']},
                    name='E_sc')

# Remove E_sc from the measured electric field
E_corrected = edi_data['E_GSE'] - E_sc
E_corrected.name = 'E_corrected'

# Magnitude of the corrected electric field
E_corrected_mag = xr.DataArray(np.linalg.norm(E_corrected,
                                              axis=E_corrected.get_axis_num('E_index')),
                               dims='Epoch',
                               coords={'Epoch': edi_data['Epoch']},
                               name='|E_corrected|')

# Corotation Electric Field
Re = 6371.2 # km

# Corotation electric field: V/km = mV/m
E_corot = (-92100 * Re
           / np.linalg.norm(mec_data['R_sc'],
                            ord=2,
                            axis=mec_data['R_sc'].get_axis_num('R_sc_index'))**2
           )

E_corot = xr.DataArray(E_corot,
                       dims='Epoch',
                       coords={'Epoch': mec_data['Epoch']},
                       name='E_corot')

# Electric field magnitude of EDI data for comparison
edi_data['|E|'] = xr.DataArray(np.linalg.norm(edi_data['E_GSE'],
                                              axis=edi_data['E_GSE'].get_axis_num('E_index')),
                               dims='Epoch',
                               coords={'Epoch': edi_data['Epoch']})

# Combine all into one data set
data = xr.merge([fgm_data, edi_data, mec_data, E_sc, E_corrected, E_corrected_mag, E_corot])

data.to_netcdf(path="data.nc")