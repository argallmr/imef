import argparse
import imef.data.data_manipulation as dm
import xarray as xr
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description='Takes data file downloaded from sample_data, calculates the convective electric field, and bins it by r, theta, and kp'
    )

    parser.add_argument('file',
                        type=str,
                        help='File that contains the data the user wants to bin')

    parser.add_argument('-nt', '--notheta',
                        help='Bin the data by only r and kp, instead of the default r, theta, and kp',
                        action='store_true')

    parser.add_argument('-nk', '--nokp',
                        help='Bin the data by only r and theta, instead of the default r, theta, and kp',
                        action='store_true')

    args = parser.parse_args()
    file = args.file
    notheta = args.notheta
    nokp = args.nokp

    if notheta == nokp == True:
        raise AttributeError('Notheta and Nokp cannot be active at the same time')

    data = xr.open_dataset(file+'.nc')

    if notheta:
        binned_data = dm.bin_kp_r_theta(data, 'E_con', dMLT=24)
        binned_data['E_con_mean'] = binned_data['E_con_mean'][:,:,0,:]
        binned_data['E_con_counts'] = binned_data['E_con_counts'][:,:,0,:]
        binned_data = binned_data.drop_vars('theta')
        binned_filename = file + '_binned_r_kp.nc'
    elif nokp:
        # By setting kp_bins to be the entire range of kp values, we get binned by only r_theta_cart
        binned_data = dm.bin_kp_r_theta(data, 'E_con', kp_bins=np.array([0, 10]))
        # But it has a Kp dimension still (that is 1 unit big). Remove that dimension from dataset and variables
        binned_data['E_con_mean'] = binned_data['E_con_mean'][0]
        binned_data['E_con_counts'] = binned_data['E_con_counts'][0]
        binned_data = binned_data.drop_vars('kp')
        binned_filename = file + '_binned_r_theta.nc'
    else:
        binned_data = dm.bin_kp_r_theta(data, 'E_con')
        binned_filename = file+'_binned_r_theta_kp.nc'

    binned_data.to_netcdf(binned_filename)



if __name__ == '__main__':
    main()