import pandas as pd
import numpy as np

df = pd.read_csv('9_10_15_to_9_16_2016.csv')

df_bins_loc = pd.read_csv('../csv/bins_loc.csv')
df_mlt = pd.read_csv('../csv/mlt_loc.csv')
df_r = pd.read_csv('../csv/r_loc.csv')

er_list = []
eaz_list = []

for i in range(0, 144):
    theta_loc = df_bins_loc['MLT_Loc'][i]
    theta_b = np.average([df_mlt['Left'][theta_loc], df_mlt['Right'][theta_loc]])
    theta = theta_b * (2 * np.pi / 24)

    r_hat = (np.cos(theta), np.sin(theta))
    phi_hat = (-np.sin(theta), np.cos(theta))

    E = (df['EDI_X'][i], df['EDI_Y'][i])
    er = np.dot(E, r_hat)
    eaz = np.dot(E, phi_hat)

    er_list.append(er)
    eaz_list.append(eaz)


df_f = pd.DataFrame(er_list, columns=["ER"])
df_f['EAZ'] = eaz_list
df_f['MLT'] = df['MLT']
df_f['L'] = df['L']

df_f.to_csv('polar_bin.csv')