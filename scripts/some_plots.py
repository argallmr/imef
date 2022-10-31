import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import imef.data.data_manipulation as dm
import visualizations.plot_nc_data as xrplot
import visualizations.visualizations as vis


def main():
    data = xr.open_dataset('mms1_imef_srvy_l2_5sec_20150901000000_20220701000000.nc')
    data_binned_nk = xr.open_dataset(
        'example_datasets/mms1_imef_srvy_l2_5sec_20150901000000_20220701000000_binned_r_theta.nc')
    data_binned = xr.open_dataset(
        'example_datasets/mms1_imef_srvy_l2_5sec_20150901000000_20220701000000_binned_r_theta_Kp.nc')

    # vis.ief_holes_hist(data, index='AL', bins = np.arange(-2000, 200, 20))

    # vis.ief_holes_hist(data)

    # vis.create_histogram(data, index='Sym-H', bins=np.arange(-140, 60, 2))

    # np.arange(0, 90, 10/3)-.1
    vis.create_histogram(data, index='Kp', bins=np.array([0,1,2,3,4,5,6,7,9]), checkmarks=[1.1])

    fig, axes = vis.plot_global_efield_one(data_binned_nk, None)
    plt.show()

    fig, axes = vis.plot_global_counts_index(data_binned, varname='E_con', index='Kp')
    plt.show()

    fig, axes = vis.plot_efield_r_index(data_binned, 'E_con', index='Kp')
    plt.show()

    vis.efield_vs_kp_plot(data_binned, 'E_con')
    plt.show()

    L = data_binned['r'][2:].values
    MLT = data_binned['theta'].values * 12 / np.pi
    imef_data = xr.Dataset(coords={'L': L, 'MLT': MLT, 'polar': ['r', 'phi'], 'cartesian': ['x', 'y', 'z']})
    imef_data['E_con_mean'] = data_binned_nk['E_con_mean'][2:]
    # POLAR
    imef_data = dm.convert_to_polar(imef_data, 'E_con_mean')
    potential = dm.calculate_potential(imef_data, 'E_con_mean_polar')
    xrplot.plot_potential(imef_data, potential)


if __name__ == '__main__':
    main()