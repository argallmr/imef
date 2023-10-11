import xarray as xr
import numpy as np
import imef.data.data_manipulation as dm
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='This program takes a data file from sample_data.py and performs random undersampling on it, '
                    'removing quiet time data until the counts of storm and quiet time data are closer together. '
                    'Note that this program does not work with create_neural_network.py, as the removed data points are required to train the NN'
    )

    parser.add_argument('input_filename', type=str,
                        help='File name of the data created by sample_data.py. Do not include file extension')

    parser.add_argument('-qsr', '--quiet_storm_ratio',
                        default=1,
                        type=float,
                        help='Ratio of quiet versus storm time data. Default is 1',
                        )

    args = parser.parse_args()
    filename=args.input_filename
    quiet_storm_ratio = args.quiet_storm_ratio

    data = xr.open_dataset(filename + '.nc')

    undersampled_data = dm.random_undersampling(data, quiet_storm_ratio=quiet_storm_ratio)

    undersampled_data.to_netcdf(filename+'_undersampled_'+str(quiet_storm_ratio)+'.nc')


if __name__ == '__main__':
    main()