import xarray as xr
import pdb
import numpy as np
import pandas as pd

data = xr.open_dataset('data.nc')

df = pd.DataFrame(data=data['E_corrected'][:,0].values, columns=["EDI_X"])

# df['EDI_X'] = data['E_corrected'][:,0].values
df['EDI_Y'] = data['E_corrected'][:,1].values
df['EDI_Z'] = data['E_corrected'][:,2].values
df['MLT'] = data['MLT_sc'].values
df['L'] = data['L_sc'].values

df.to_pickle('data.pkl')