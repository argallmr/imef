import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import plot_nc_data as xrplot
import argparse

# This function makes a bunch of Kp related plots. Yeah

def main():
    parser = argparse.ArgumentParser(
        description='Make a bunch of Kp-related plots'
    )

    parser.add_argument('data_file', type=str, help='Name of file that contains electric field data. The Kp bin sizes should either be 1 or 3. If they are 1, use the argument -one')

    parser.add_argument('-one', '--one', help='The data is in Kp bin sizes of 1. Default is 3',
                        action='store_true')

    parser.add_argument('-p', '--polar', help='Data from data_file is in polar coordinates. Default is cartesian',
                        action='store_true')

    # This could be a nice thing to add. but that being said I would have to go to every function I use in this and implement that
    # parser.add_argument('-s', '--save', help='Save the plots as pdfs. Names will be creating_kp_plots_(1,2,...).pdf',
    #                     action='store_true')

    args = parser.parse_args()

    # Designate arguments
    filename = args.data_file
    one=args.one
    polar = args.polar

    if polar:
        mode='polar'
    else:
        mode='cartesian'

    data = xr.open_dataset(filename)

    # If the user gave Kp bin sizes of one or three
    if one:
        xrplot.line_plot(data, mode=mode, MLT_range=[3, 9])
        xrplot.line_plot(data, mode=mode, MLT_range=[15, 21])
    else:
        xrplot.efield_vs_kp_plot(data, mode=mode)



if __name__ == '__main__':
    main()
