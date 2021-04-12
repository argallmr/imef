import xarray as xr
import pdb
import numpy as np
import pandas as pd

df = pd.read_pickle('train_data.pkl')

print(df)




#data = xr.open_dataset('data.nc')

#df = pd.DataFrame(data=data['E_GSE'][:,0].values, columns=["EDI_X"])

#df['IsEdi'] = df['EDI_X'].apply(lambda x: 1 if np.abs(x) > 0 else 0)

#df.to_pickle('data.pkl')