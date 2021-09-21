import xarray as xr
import plot_nc_data as xrplot
import data_manipulation as dm

def main():
    # Open Data
    imef_data = xr.open_dataset('6years_store_efield_data.nc')

    # Select Desired Range of Data
    newtest = imef_data.where(imef_data.iL < 11, drop=True)
    imef_data = newtest.where(newtest.iL > 3, drop=True)

    # Find the range of L values used, so we can plot over the proper range of values
    min_Lvalue = imef_data['L'][0, 0].values
    max_Lvalue = imef_data['L'][-1, 0].values
    nL = int(max_Lvalue - min_Lvalue+1)
    nMLT = 24

    # Plot Electric Field + Count Data
    #xrplot.plot_efield_polar(nL, nMLT, imef_data)

    # Calculate Potential
    V = dm.calculate_potential(imef_data)

    # Plot Potential
    xrplot.plot_potential(nL, nMLT, imef_data, V)

if __name__ == '__main__':
    main()
