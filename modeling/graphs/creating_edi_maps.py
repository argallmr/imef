import xarray as xr
import plot_nc_data as xrplot
import data_manipulation as dm
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Calculate the electric potential from electric field data given by the user (from a file created by store_efield_data), and exports to a csv file '
                    'Can also plot the electric field data and electric potential. '
                    'Cartesian data can still be used, however it will only be plotted, and no potential values will be calculated'
    )

    parser.add_argument('data_file', type=str, help='Name of file that contains electric field data')

    parser.add_argument('data_name', type=str, help='Name of variable in the data file that contains electric field data. eg: E_GSE_mean')

    parser.add_argument('output_file_name', type=str, help='Name of the newly created file containing the potential values. Do not include file extension')

    parser.add_argument('-n', '--no-show', help='Do not create and display the plots of the electric field and electric potential. Default is to plot', action='store_true')

    parser.add_argument('-p', '--polar',help='Data from data_file is in polar coordinates. Default is cartesian', action='store_true')

    args = parser.parse_args()

    # Designate arguments
    filename=args.data_file
    variable_name=args.data_name
    new_filename=args.output_file_name
    no_show=args.no_show
    polar=args.polar

    # Open data file
    imef_data = xr.open_dataset(filename)

    # Find the range of L values used in the given data
    min_Lvalue = imef_data['L'][0, 0].values
    max_Lvalue = imef_data['L'][-1, 0].values

    # Find the number of bins in the radial and azimuthal directions. The azimuthal should be in MLT, meaning it will always be 24
    nL = int(max_Lvalue - min_Lvalue+1)
    nMLT = 24

    # Calculate Potential and save to a csv file with a name given by the user
    # Note that if you want an accurate electric potential you must use polar coordinates when running store_efield_data, which is why it is not run when cartesian values are inputted
    if polar==True:
        V = dm.calculate_potential(imef_data, variable_name)
        np.savetxt(new_filename+".csv", V, delimiter=",")

    # plotting (if the user wants)
    if no_show==False and polar==True:
        # Plot Electric Field + Count Data
        xrplot.plot_efield_polar(nL, nMLT, imef_data, variable_name, log=True)
        # Plot Potential
        xrplot.plot_potential(nL, nMLT, imef_data, V)
    elif no_show==False and polar==False:
        # Plot Electric Field + Count Data
        xrplot.plot_efield_cartesian(nL, nMLT, imef_data, variable_name, log=True)


if __name__ == '__main__':
    main()
