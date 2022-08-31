import xarray as xr
import numpy as np
import imef.data.data_manipulation as dm
import argparse


def reverse_bins(bins, length_of_data):
    reversed_bins = []
    counter = 0

    while counter < len(bins):
        if counter == 0:
            # put first bin in list (if needed)
            if bins[0][0] == 0:
                pass
            else:
                reversed_bins.append([0, bins[0][0]])
        else:
            # Put reversed bins in list
            reversed_bins.append([bins[counter - 1][1], bins[counter][0]])
        counter += 1

    # put last bin in list (if needed)
    if bins[-1][1] != length_of_data:
        reversed_bins.append([bins[-1][1], length_of_data])

    return reversed_bins


# only works for dst right now
def random_undersampling(data, threshold=-40, quiet_storm_ratio=1.0):
    # Tbh not quite sure if I understood the algorithm correctly. It does what I think it wants, so I'll stick with it for now.
    # To be clear, this definitely undersamples the data. However the way I do it is by reducing the number of datapoints in each bin by a certain percentage. The authors may not do the same
    # But here is the link in case it's needed: https://link.springer.com/article/10.1007/s41060-017-0044-3

    intervals_of_storm_data = dm.get_storm_intervals(data, threshold=threshold)
    storm_counts=0
    for start, end in intervals_of_storm_data:
        storm_counts += end-start
    quiet_counts=len(data['time'])-storm_counts

    if quiet_counts > storm_counts:
        bins_to_undersample = reverse_bins(intervals_of_storm_data, len(data['time']))
        percent_to_reduce = quiet_storm_ratio * storm_counts / quiet_counts

        if percent_to_reduce >= 1:
            raise ValueError('quiet_storm_ratio is too large. The max value for this dataset is '+str(quiet_counts/storm_counts))
        elif percent_to_reduce <= 0:
            raise ValueError('quiet_storm ratio is too small. It must be greater than 0')

        all_times = []
        for start, end in intervals_of_storm_data:
            all_times.append(data['time'].values[start:end])
        for start, end in bins_to_undersample:
            new_times_in_bin = np.random.choice(data['time'][start:end], int(percent_to_reduce * (end - start)), replace=False)
            all_times.append(new_times_in_bin)
        all_times = np.concatenate(all_times)
        all_times = np.sort(all_times)

        undersampled_data = data.sel(time=all_times)

        return undersampled_data
    else:
        # I don't know if a) this will ever come up, or b) if this did come up, we would want to undersample the storm data. So raising error for now
        raise Warning('There is more storm data than quiet data. Skipping undersampling.')


def main():
    parser = argparse.ArgumentParser(
        description='PUT DESCRIPTION HERE'
    )

    parser.add_argument('input_filename', type=str,
                        help='File name of the data created by sample_data.py. Do not include file extension')

    parser.add_argument('-r', '--quiet_storm_ratio',
                        default=1,
                        type=float,
                        help='Ratio of quiet versus storm time data. Default is 1',
                        )

    args = parser.parse_args()
    filename=args.input_filename
    quiet_storm_ratio = args.quiet_storm_ratio

    data = xr.open_dataset(filename + '.nc')

    undersampled_data = random_undersampling(data, quiet_storm_ratio=quiet_storm_ratio)

    undersampled_data.to_netcdf(filename+'_undersampled_'+str(quiet_storm_ratio)+'.nc')


if __name__ == '__main__':
    main()