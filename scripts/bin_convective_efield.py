import argparse
import imef.data.data_manipulation as dm
import xarray as xr

def main():
    parser = argparse.ArgumentParser(
        description='Takes data file downloaded from sample_data, calculates the convective electric field, and bins it by r, theta, and kp'
    )

    parser.add_argument('file',
                        type=str,
                        help='File that contains the data the user wants to bin')

    args = parser.parse_args()
    file = args.file

    data = xr.open_dataset(file+'.nc')

    # Note that E_con is NOT the convective electric field, but rather the spacecraft electric field. Should probably be edited at some point
    E_convective = data['E_EDI'] - data['E_cor'] - data['E_con']
    data['E_convective'] = E_convective

    binned_data = dm.bin_kp_r_theta(data, 'E_convective')
    binned_filename = file+'_binned_r_theta_kp.nc'

    binned_data.to_netcdf(binned_filename)



if __name__ == '__main__':
    main()