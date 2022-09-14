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

    parser.add_argument('-i', '--index',
                        type=str,
                        default='Kp',
                        help='Geomagnetic index to bin by')

    parser.add_argument('-b', '--bins', nargs='+',
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 9],
                        help='The list of bins that you want to use. e.g. -b 1 3 5 7 9 results in using [1,3), [3,5), ...')

    parser.add_argument('-nt', '--notheta',
                        help='Bin the data by only r and kp, instead of the default r, theta, and kp',
                        action='store_true')

    parser.add_argument('-ni', '--noindex',
                        help='Bin the data by only r and theta, instead of the default r, theta, and kp',
                        action='store_true')

    args = parser.parse_args()
    file = args.file
    index = args.index
    bins = args.bins
    notheta = args.notheta
    noindex = args.noindex

    bins = np.array(bins).astype(int)

    if notheta == noindex == True:
        raise AttributeError('Notheta and Nokp cannot be active at the same time')

    data = xr.open_dataset(file + '.nc')

    if notheta:
        binned_data = dm.bin_index_r_theta(data, 'E_con', index=index, dMLT=24, index_bins=bins)
        binned_data['E_con_mean'] = binned_data['E_con_mean'][:, :, 0, :]
        binned_data['E_con_counts'] = binned_data['E_con_counts'][:, :, 0, :]
        binned_data = binned_data.drop_vars('theta')
        binned_filename = file + '_binned_r_' + index + '.nc'
    elif noindex:
        # By setting kp_bins to be the entire range of kp values, we get binned by only r_theta_cart
        binned_data = dm.bin_index_r_theta(data, 'E_con', index=index, index_bins=np.array([min(data[index].values)-1, max(data[index].values)+1]))
        # But it has a Kp dimension still (that is 1 unit big). Remove that dimension from dataset and variables
        binned_data['E_con_mean'] = binned_data['E_con_mean'][0]
        binned_data['E_con_counts'] = binned_data['E_con_counts'][0]
        binned_data = binned_data.drop_vars(index)
        binned_filename = file + '_binned_r_theta.nc'
    else:
        binned_data = dm.bin_index_r_theta(data, 'E_con', index=index, index_bins=bins)
        binned_filename = file + '_binned_r_theta_' + index + '.nc'

    binned_data.to_netcdf(binned_filename)


if __name__ == '__main__':
    main()
