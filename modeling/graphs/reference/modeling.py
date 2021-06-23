import pandas as pd
import numpy as np

# Import formatted data
df = pd.read_pickle('data.pkl')

print(df)

# Create the bins
mlt_bins = [(i, i+1) for i in range(24)]
r_bins = [(i, i+1) for i in range(1, 8)]

# Format to work with pandas
pd_mlt_bins = pd.IntervalIndex.from_tuples(mlt_bins)
pd_r_bins = pd.IntervalIndex.from_tuples(r_bins)

# Cut into bins
df['BIN_MLT'] = pd.cut(df['MLT'], pd_mlt_bins)
df['BIN_R'] = pd.cut(df['L'], pd_r_bins)

df_bins_loc = pd.read_csv('../csv/bins_loc.csv')
df_mlt = pd.read_csv('../csv/mlt_loc.csv')
df_r = pd.read_csv('../csv/r_loc.csv')

means = []

# i = bin number
for i in range(0, 144):
  loc_mlt = df_bins_loc['MLT_Loc'][i]
  loc_r = df_bins_loc['R_Loc'][i]

  df_bin1 = df[df['BIN_MLT'] == pd.Interval(left=df_mlt['Left'][loc_mlt], right=df_mlt['Right'][loc_mlt])]

  df_bin1 = df_bin1[df_bin1['BIN_R'] == pd.Interval(left=df_r['Left'][loc_r], right=df_r['Right'][loc_r])]

  means.append([df_bin1.mean()['EDI_X'], df_bin1.mean()['EDI_Y'], df_bin1.mean()['EDI_Z'],
                df_bin1.mean()['MLT'], df_bin1.mean()['L']])

df_m = pd.DataFrame(means, columns=['EDI_X', 'EDI_Y', 'EDI_Z', 'MLT', 'L'])

df_m.to_csv('9_10_15_to_9_16_2016.csv')