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

    # Note that E_con is NOT the convective electric field, but rather the spacecraft electric field
    E_convective = data['E_EDI'] - data['E_cor'] - data['E_con']
    data['E_convective'] = E_convective

    if notheta:
        counts, avg, kp_bins, r_bins = dm.bin_r_kp(data, 'E_convective')
        r_bins=np.arange(0, 10)
        kp_bins=np.arange(0, 8)
        binned_data = xr.Dataset(coords={'Kp': kp_bins, 'L':r_bins})
        binned_data['E_convective_counts']=xr.DataArray(counts, dims=['Kp', 'L'], coords={'Kp':kp_bins, 'L': r_bins})
        binned_data['E_convective_mean'] = xr.DataArray(avg, dims=['Kp', 'L'], coords={'Kp': kp_bins, 'L': r_bins})
        binned_filename = file + '_binned_r_kp.nc'
    elif nokp:
        # By setting kp_bins to be the entire range of kp values, we get binned by only r_theta_cart
        binned_data = dm.bin_kp_r_theta(data, 'E_convective', kp_bins=np.array([0, 10]))
        # But it has a Kp dimension still (that is 1 unit big). Remove that dimension from dataset and variables
        binned_data['E_convective_mean'] = binned_data['E_convective_mean'][0]
        binned_data['E_convective_counts'] = binned_data['E_convective_counts'][0]
        binned_data = binned_data.drop_vars('kp')
        binned_data = binned_data.rename({'comp':'cart'})
        binned_filename = file + '_binned_r_theta.nc'
    else:
        binned_data = dm.bin_kp_r_theta(data, 'E_convective')
        binned_data = binned_data.rename({'comp': 'cart'})
        binned_filename = file+'_binned_r_theta_kp.nc'

    binned_data.to_netcdf(binned_filename)



if __name__ == '__main__':
    main()