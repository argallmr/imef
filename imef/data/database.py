import datetime as dt
import xarray as xr
import numpy as np
from scipy.stats import binned_statistic_2d
import torch
from pathlib import Path
from warnings import warn
import os

import imef.data.download_data as dd
import imef.data.data_manipulation as dm

from pymms.data.util import NoVariablesInFileError

data_dir = Path('~/data/imef/').expanduser()
if not data_dir.exists():
    data_dir.mkdir(parents=True)

R_E = 6378.1  # km


def bin_files(r_range=(3, 10), theta_range=(0, 360), dr=1, dtheta=15):
    files = [data_dir / 'mms4_imef_srvy_l2_5sec_20150915000000_20160101000000.nc',
             data_dir / 'mms4_imef_srvy_l2_5sec_20160101000000_20170101000000.nc',
             data_dir / 'mms4_imef_srvy_l2_5sec_20170101000000_20180101000000.nc',
             data_dir / 'mms4_imef_srvy_l2_5sec_20180101000000_20190101000000.nc']

    # Define the grid extent and spacing in the equatorial plane
    theta = np.deg2rad(np.arange(theta_range[0], theta_range[1] + dtheta, dtheta))
    r = np.arange(r_range[0], r_range[1] + dr, dr)

    ds_out = {}

    for file in files:
        data = (xr.load_dataset(file)
                .drop(['component', 'E_index'])
                .rename({'component': 'cart',
                         'E_index': 'cart'})
                )

        # Transform the data into the corotating frame
        corot_data = corotating_frame(data)

        # Convert cartesian coordinates to cylindrical
        r_cyl = dm.cart2cyl(data['R_sc'])
        r_pos = r_cyl.loc[:, 'r'] / R_E
        theta_pos = (2 * np.pi + r_cyl.loc[:, 'phi']) % 2 * np.pi

        # Bin the data
        ds_temp = bin_ds(r_pos, theta_pos, corot_data, r, theta)

        # Combine data across files
        if len(ds_out) == 0:
            ds_out = ds_temp
        else:
            ds_out = combine_binned_results(ds_out, ds_temp)

    # Create file name
    #   - Add "-binned" to optional descriptor
    #   - Take start time from first file
    #   - Take end time from last file
    parts = str(files[0].name).split('_')
    fname = (data_dir / '_'.join((*parts[0:4], parts[4] + '-binned', parts[5],
                                  str(files[-1].name).split('_')[6][:-3]))
             ).with_suffix('.nc')

    # Write to file
    ds_out.to_netcdf(path=fname)

    return fname


def bin_ds(r_pos, theta_pos, ds, r, theta):
    def bin_var(r_pos, theta_pos, data_to_bin, statistic):
        # Bin the data, calculating the average
        stat, x_edge, y_edge, num = binned_statistic_2d(x=r_pos,
                                                        y=theta_pos,
                                                        values=data_to_bin,
                                                        statistic=statistic,
                                                        bins=[r, theta])
        return np.where(np.isnan(stat), 0, stat)

    vars_out = {}

    # Loop over each variable in the dataset
    for name, var in ds.items():
        if not name.startswith('E_'):
            continue

        # A time series
        if var.ndim == 1:
            if np.issubdtype(var.dtype, np.timedelta64):
                temp = bin_var(var, 'mean')(var.data.astype(float))
                temp = temp.astype(np.timedelta64)
            else:
                temp = bin_var('mean')

            temp = xr.DataArray(temp,
                                dims=('r', 'theta'),
                                coords={'r': r[:-1],
                                        'theta': theta[:-1]},
                                name=name)

        # A vector of time series
        elif var.ndim == 2:
            temp_vars = []

            # Bin each component of the vector
            isgood = ~np.isnan(var.data[:, 0])
            counts = bin_var(r_pos.data[isgood], theta_pos.data[isgood],
                             var.data[isgood, 0], 'count')
            counts = xr.DataArray(counts,
                                  dims=('r', 'theta'),
                                  coords={'r': r[:-1],
                                          'theta': theta[:-1]},
                                  name=name + '_count')

            for idx in range(var.shape[1]):
                temp_binned = bin_var(r_pos.data[isgood], theta_pos.data[isgood],
                                      var.data[isgood, idx], 'mean')
                temp_vars.append(temp_binned)

            # Combine bin-averaged components into an array
            temp = xr.DataArray(np.stack(temp_vars, axis=2),
                                dims=('r', 'theta', var.dims[1]),
                                coords={'r': r[:-1],
                                        'theta': theta[:-1],
                                        var.dims[1]: var.coords[var.dims[1]]},
                                name=name)

        else:
            warn('Variable {0} has > 2 dimensions. Skipping'
                 .format(name))

        # Save the variables in the output dictionary
        vars_out[name] = temp
        vars_out[name + '_count'] = counts

    # Create a new dataset
    return xr.Dataset(vars_out)


def combine_binned_results(ds_old, ds_new):
    '''
    Take the weighted average of to datasets binned by `bin_files`.

    Parameters
    ----------
    ds_old, ds_new : `xarray.Dataset`
        The two datasets to be combined together. The weighted average
        is taken between the old and new variables. Counts per bin are
        summed.

    Returns
    -------
    ds : `xarray.Dataset`
        A new dataset with averaged data and updated counts per bin.
    '''

    for name, var in ds_old.items():
        if name.endswith('count'):
            continue

        # Weighted average of E-field
        ds_old[name] = ((ds_old[name + '_count'] * ds_old[name]
                         + ds_new[name + '_count'] * ds_new[name]
                         ) / (ds_old[name + '_count'] + ds_new[name + '_count'])
                        ).where(~np.isnan(ds_old[name]), other=0)

        # Sum of counts
        ds_old[name + '_count'] += ds_new[name + '_count']

    return ds_old


def combine_intervals(files):
    '''
    Combine time series files created by `multi_interval` or `one_interval`
    into a single dataset.

    Parameters
    ----------
    files : list of str
        Paths to the files to be read and combined

    Returns
    -------
    fname : str
        Name of the combined output file. File name is the same as the input
        file names, but with the start time taken from the first file and the
        end time taken from the last file.
    '''

    # Read each file separately
    data = []
    for file in files:
        temp = xr.load_dataset(file)
        temp = (temp.drop(('E_index', 'component'))
                .rename({'E_index': 'cart',
                         'component': 'cart'})
                )

        data.append(temp)

    # Merge all files into a single dataframe
    data = xr.concat(data, dim='time')

    # Create an output file name
    fstart = Path(files[0]).stem.split('_')
    fstop = Path(files[-1]).stem.split('_')
    fname = (data_dir / '_'.join((*fstart[0:6], fstop[6]))).with_suffix('.nc')

    # Write to file
    data.to_netcdf(fname)

    return fname


def multi_interval(sc, mode, level, t0, t1,
                   dt_out=dt.timedelta(seconds=5)):
    '''
    Save data required by the IMEF model to a netCDF file.

    Given a time interval [t0, t1), break the interval into chunks based on the
    data archive system (1-day), download each chunk, combine chunks together,
    and save to a file

    Parameters
    ----------
    sc : str
        Spacecraft identifier (mms1, mms2, mms3, mms4)
    mode : str
        Data rate mode (srvy, brst)
    level : str
        Data level (l1a, l2)
    t0, t1 : `datetime.datetime`
        Start and end times of the data interval to download
    dt_out : `datetime.timedelta`, default=5s
        Sampling interval to which the data will be resampled

    Returns
    -------
    fpath : str
        Path to the output file.
    '''

    data = []
    intervals = []
    problem_intervals = []
    problem_msgs = []

    # Break the interval into daily subintervals
    #   - MMS data files are daily files.
    #   - Prevent two files from being downloaded by ending the subinterval
    #     one microsecond before midnight
    t_inc = dt.timedelta(days=1)
    t_start = t0
    t_end = t0 + (dt.datetime.combine(t0.date()
                                      + dt.timedelta(days=1),
                                      dt.time(0)) - t0
                  ) - dt.timedelta(microseconds=1)

    # Loop over each subinterval
    while t_start < t1:

        # Do not download data past the end of the interval
        if t_end > t1:
            t_end = t1
        intervals.append((t_start, t_end))

        print('({0}, {1})'.format(t_start, t_end))

        # Download one day
        try:
            data.append(one_interval(sc, mode, level, t_start, t_end,
                                     dt_out=dt_out))
        except NoVariablesInFileError:
            # If there is no data in the mms files, get the index data, and get the edi, fgm, etc but with nans for all the values
            data.append(one_index_interval(sc, mode, level, t_start, t_end,
                                     dt_out=dt_out, nans=True))
        except Exception as E:
            # raise E
            print('Error during interval {0} - {1}'
                  .format(t_start, t_end))
            print(type(E))
            print(E)
            problem_intervals.append((t_start, t_end))
            problem_msgs.append(str(E))

        # Continue to the next interval
        t_start = dt.datetime.combine(t_start.date(), dt.time(0)) + t_inc
        t_end += t_inc

    # Combine all of the datasets together
    ds = xr.concat(data, dim='time')

    # Fill in some attributes

    # Inputs
    #   - It appears 'mode' cannot be an attribute when writing to netCDF
    ds.attrs['sc'] = sc
    ds.attrs['data_rate_mode'] = mode
    ds.attrs['level'] = level
    ds.attrs['start_date'] = t0.strftime('%Y-%m-%dT%H:%M:%S')
    ds.attrs['end_date'] = t1.strftime('%Y-%m-%dT%H:%M:%S')
    ds.attrs['resample'] = '{0:d} s'.format(int(dt_out.total_seconds()))

    # Data intervals
    ds.attrs['n_intervals'] = len(intervals)
    ds.attrs['interval_start'] = [interval[0].strftime('%Y-%m-%dT%H:%M:%S')
                                  for interval in intervals]
    ds.attrs['interval_stop'] = [interval[1].strftime('%Y%m%d%H%M%S')
                                 for interval in intervals]

    # Errors
    ds.attrs['n_bad_intervals'] = len(problem_intervals)
    ds.attrs['bad_int_start'] = [interval[0].strftime('%Y-%m-%dT%H:%M:%S')
                                 for interval in problem_intervals]
    ds.attrs['bad_int_stop'] = [interval[1].strftime('%Y-%m-%dT%H:%M:%S')
                                for interval in problem_intervals]
    ds.attrs['bad_int_msgs'] = problem_msgs

    # netCDF files do not like boolean values
    #   - Convert to int8
    #   - **** 5/28/2022: This has been fixed in the GitHub version of PYMMS ****
    for name, var in ds.data_vars.items():
        try:
            var.attrs['rec_vary'] = np.int8(var.attrs['rec_vary'])
        except KeyError:
            pass
    for name, var in ds.coords.items():
        try:
            var.attrs['rec_vary'] = np.int8(var.attrs['rec_vary'])
        except KeyError:
            pass

    # Write the dataset to a file
    #   - Format the file name like standard MMS files
    #   - Include the sample interval in the optional descriptor

    # Optional descriptor in minutes and seconds: XminYsec
    sec = dt_out.total_seconds()
    if dt_out >= dt.timedelta(seconds=60):
        mnt = int(sec / 60)
        sec = sec % 60
        optdesc = '{0:d}min{1:d}sec'.format(mnt, int(sec))
    else:
        optdesc = '{0:d}sec'.format(int(sec))

    # File path
    fname = 'do_not_use'
    fpath = (data_dir / fname).with_suffix('.nc')

    # Write to file
    ds.to_netcdf(path=fpath)

    # For some reason MLT sometimes gets changed to nanosecond timedelta64 objects in the above .to_netcdf
    # it can be fixed by opening that file, changing it back to hours, and outputting to a file
    # it is MLT for dt_out=5 seconds, it does it for dt_out=10 seconds, but not for dt_out=60 seconds
    # I then remove the first file, since it is incorrect and there is no use for it
    data = xr.open_dataset(fpath)

    if type(data['MLT'].values[0]) == type(np.timedelta64(1, 'ns')):
        data['MLT'] = data['MLT']/np.timedelta64(1, 'h')

    fname2 = '_'.join((sc, 'imef', mode, level, optdesc,
                      t0.strftime('%Y%m%d%H%M%S'),
                      t1.strftime('%Y%m%d%H%M%S')))
    fpath2 = (data_dir / fname2).with_suffix('.nc')

    data.to_netcdf(path=fpath2)

    data.close()
    os.remove(fpath)

    return fpath2


def one_interval(sc, mode, level, t0, t1, dt_out=None):
    '''
    Download and read data required by the IMEF model to a netCDF file.

    Parameters
    ----------
    sc : str
        Spacecraft identifier (mms1, mms2, mms3, mms4)
    mode : str
        Data rate mode (srvy, brst)
    level : str
        Data level (l1a, l2)
    t0, t1 : `datetime.datetime`
        Start and end times of the data interval to download
    dt_out : `datetime.timedelta`, default=5s
        Sampling interval to which the data will be resampled

    Returns
    -------
    data : `xarray.Dataset`
        The requested data.
    '''

    # Get the primary data products
    edi_data = dd.get_data(sc, 'edi', mode, level, t0, t1, dt_out=dt_out)
    fgm_data = dd.get_data(sc, 'fgm', mode, level, t0, t1, dt_out=dt_out)
    mec_data = dd.get_data(sc, 'mec', mode, level, t0, t1, dt_out=dt_out)
    dis_data = dd.get_data(sc, 'dis', mode, level, t0, t1, dt_out=dt_out)
    des_data = dd.get_data(sc, 'des', mode, level, t0, t1, dt_out=dt_out)
    edp_data = dd.get_data(sc, 'edp', mode, level, t0, t1, dt_out=dt_out)
    scpot_data = dd.get_data(sc, 'scpot', mode, level, t0, t1, dt_out=dt_out)
    kp_data = dd.get_kp_data_v2(t0, t1, dt_out=dt_out)
    dst_data = dd.get_dst_data(t0, t1, dt_out=dt_out)
    omni_data = dd.get_omni_data(t0, t1, dt_out=dt_out)
    # orbit_data = dd.get_orbit_number(sc, t0, t1, dt_out=dt_out)

    # Get the corotation electric field
    E_cor = dm.corotation_electric_field(mec_data['R_sc'])

    # Get the spacecraft electric field
    E_sc = dm.E_convection(-mec_data['V_sc'], fgm_data['B_GSE'][:, 0:3])

    E_con = edi_data['E_GSE'] - E_cor - E_sc

    # Calculate the plasma convection field
    E_DIS = dm.E_convection(dis_data['V_DIS'],
                            fgm_data['B_GSE'][:, 0:3].reindex_like(dis_data['V_DIS']))
    E_DES = dm.E_convection(des_data['V_DES'],
                            fgm_data['B_GSE'][:, 0:3].reindex_like(des_data['V_DES']))

    # Create a dataset
    return xr.Dataset({'E_EDI': edi_data['E_GSE'],
                       'B_GSE': fgm_data['B_GSE'],
                       'E_cor': E_cor,
                       'E_sc': E_sc,
                       'E_con': E_con,
                       'E_DIS': E_DIS,
                       'E_DES': E_DES,
                       'E_EDP': edp_data['E_GSE'],
                       'Kp': kp_data,
                       'Dst': dst_data,
                       'Sym-H': omni_data['Sym-H'],
                       'AE': omni_data['AE'],
                       'AL': omni_data['AL'],
                       'AU': omni_data['AU'],
                       'IEF': omni_data['IEF'],
                       'Scpot': scpot_data,
                       'R_sc': mec_data['R_sc'],
                       'V_sc': mec_data['V_sc'],
                       'L': mec_data['L'],
                       'MLT': mec_data['MLT']})


def predict_efield_and_potential(model, time=None, data=None, return_pred = True, number_of_inputs = 1):
    # A function that will take a model created by create_neural_network.py, and either a time or data argument,
    # and calculate the electric field and electric potential, plot them (if the user wants), and return the predicted values (if the user wants)

    # DOES NOT HAVE SYM-H FUNCTIONALITY

    # can input either the data you have that corresponds to the time you want to predict (aka the 5 hours of data, with the last input being the time you want to predict)
    # or you can input the time and the data will be downloaded for you
    # if both are given, the data will have priority
    if data is not None:
        complete_data = data
    elif time is not None:
        # we need the data from the 5 hours before the time to the time given
        # But the binned argument requires 1 day of data. so I do this instead
        ti = time - dt.timedelta(hours=5)
        te = time + dt.timedelta(minutes=5)

        mec_data = dd.get_mec_data('mms1', 'srvy', 'l2', ti, te, binned=True)
        kp_data = dd.get_kp_data(ti, te, expand=mec_data['time'].values)
        dst_data = dd.get_dst_data(ti, te, expand=mec_data['time'].values)

        complete_data = xr.merge([mec_data, kp_data, dst_data])
    elif time is None and data is None:
        raise TypeError('Either the desired time or the appropriate data must be given')

    test_inputs = dm.get_NN_inputs(complete_data, remove_nan=False, get_target_data=False)

    base_kp_values = test_inputs[-1].clone()

    size_of_input_vector=60*number_of_inputs+3

    for L in range(4, 11):
        for MLT in range(0, 24):
            new_row = base_kp_values.clone()
            new_row[-3] = L
            new_row[-2] = np.cos(np.pi/12*MLT)
            new_row[-1] = np.sin(np.pi/12*MLT)
            even_newer_row = torch.empty((1, size_of_input_vector))
            even_newer_row[0] = new_row
            if L == 4 and MLT == 0:
                all_locations = even_newer_row
            else:
                all_locations = torch.cat((all_locations, even_newer_row))

    model.eval()
    with torch.no_grad():
        pred = model(all_locations)

    nL = 7
    nMLT = 24

    # Create a coordinate grid
    something = np.arange(0, 24)
    another_thing = np.concatenate((something, something, something, something, something, something, something)).reshape(nL, nMLT)
    phi = (2 * np.pi * another_thing / 24)
    r = np.repeat(np.arange(4, 11), 24).reshape(nL, nMLT)

    # Start calculating the potential
    L = xr.DataArray(r, dims=['iL', 'iMLT'])
    MLT = xr.DataArray(another_thing, dims=['iL', 'iMLT'])

    # create an empty dataset and insert the predicted cartesian values into it. the time coordinate is nonsensical, but it needs to be time so that rot2polar works
    imef_data = xr.Dataset(coords={'L': L, 'MLT': MLT, 'polar': ['r', 'phi'], 'cartesian': ['x', 'y', 'z']})
    testing_something = xr.DataArray(pred, dims=['time', 'cartesian'], coords={'time': np.arange(0, 168), 'cartesian': ['x', 'y', 'z']})

    pred=pred.reshape(nL, nMLT, 3)

    # Create another dataset containing the locations around the earth as variables instead of dimensions
    imef_data['predicted_efield'] =xr.DataArray(pred, dims=['iL', 'iMLT', 'cartesian'],coords={'L': L, 'MLT': MLT})
    imef_data['R_sc'] = xr.DataArray(np.stack((r, phi), axis=-1).reshape(nL*nMLT, 2), dims=['time', 'polar'], coords={'time': np.arange(0,168), 'polar':['r', 'phi']})

    pred.reshape(nL * nMLT, 3)

    # have to make sure that this actually works correctly. cause otherwise imma be getting some bad stuff
    # Convert the predicted cartesian values to polar
    imef_data['predicted_efield_polar'] = dm.rot2polar(testing_something, imef_data['R_sc'], 'cartesian').assign_coords({'polar': ['r', 'phi']})

    # reshape the predicted polar data to be in terms of L and MLT, and put them into the same dataset
    somethingboi = imef_data['predicted_efield_polar'].values.reshape(nL, nMLT, 2)
    imef_data['predicted_efield_polar_iLiMLT'] = xr.DataArray(somethingboi, dims=['iL', 'iMLT', 'polar'],coords={'L': L, 'MLT': MLT})

    potential = dm.calculate_potential(imef_data, 'predicted_efield_polar_iLiMLT')

    if return_pred == True:
        return imef_data, potential


def one_index_interval(sc, mode, level, t0, t1, dt_out=None, nans=False):
    '''
    Download and read data required by the IMEF model to a netCDF file.

    Parameters
    ----------
    sc : str
        Spacecraft identifier (mms1, mms2, mms3, mms4)
    mode : str
        Data rate mode (srvy, brst)
    level : str
        Data level (l1a, l2)
    t0, t1 : `datetime.datetime`
        Start and end times of the data interval to download
    dt_out : `datetime.timedelta`, default=5s
        Sampling interval to which the data will be resampled

    Returns
    -------
    data : `xarray.Dataset`
        The requested data.
    '''

    kp_data = dd.get_kp_data_v2(t0, t1, dt_out=dt_out)
    dst_data = dd.get_dst_data(t0, t1, dt_out=dt_out)
    omni_data = dd.get_omni_data(t0, t1, dt_out=dt_out)

    if nans==False:
        return xr.Dataset({'Kp': kp_data,
                    'Dst': dst_data,
                    'Sym-H': omni_data['Sym-H'],
                    'AE': omni_data['AE'],
                    'AL': omni_data['AL'],
                    'AU': omni_data['AU'],
                    'IEF': omni_data['IEF']})

    else:
        # THIS MAY NOT BE THE BEST WAY TO DO IT. BUT ITS HOW IM GONNA DO IT FOR NOW
        # IN THE FUTURE IT WOULD BE BETTER TO NOT DOWNLOAD THIS DATA AND INSTEAD MAKE THE DATASETS FROM SCRATCH
        # These data points download data for a specific day that I know has data in it.
        example_t0 = dt.datetime(2015, 9, 10)
        example_t1 = dt.datetime(2015, 9, 11)
        example_mode = 'srvy'
        example_level = 'l2'
        example_edi_data = dd.get_data('mms1', 'edi', example_mode, example_level, example_t0, example_t1, dt_out=dt_out)
        example_fgm_data = dd.get_data('mms1', 'fgm', example_mode, example_level, example_t0, example_t1, dt_out=dt_out)
        example_mec_data = dd.get_data('mms1', 'mec', example_mode, example_level, example_t0, example_t1, dt_out=dt_out)
        example_dis_data = dd.get_data('mms1', 'dis', example_mode, example_level, example_t0, example_t1, dt_out=dt_out)
        example_des_data = dd.get_data('mms1', 'des', example_mode, example_level, example_t0, example_t1, dt_out=dt_out)
        example_edp_data = dd.get_data('mms1', 'edp', example_mode, example_level, example_t0, example_t1, dt_out=dt_out)
        example_scpot_data = dd.get_data('mms1', 'scpot', example_mode, example_level, example_t0, example_t1, dt_out=dt_out)

        # Replace all the values with nan for each of these variables, since we don't have them for these times
        example_edi_data['E_GSE'].values = np.full_like(example_edi_data['E_GSE'], np.nan)
        example_fgm_data['B_GSE'].values = np.full_like(example_fgm_data['B_GSE'], np.nan)
        example_mec_data['L'].values = np.full_like(example_mec_data['L'], np.nan)
        example_mec_data['MLT'].values = np.full_like(example_mec_data['MLT'], np.nan)
        example_mec_data['R_sc'].values = np.full_like(example_mec_data['R_sc'], np.nan)
        example_mec_data['V_sc'].values = np.full_like(example_mec_data['V_sc'], np.nan)
        example_dis_data['V_DIS'].values = np.full_like(example_dis_data['V_DIS'], np.nan)
        example_des_data['V_DES'].values = np.full_like(example_des_data['V_DES'], np.nan)
        example_edp_data['E_GSE'].values = np.full_like(example_edp_data['E_GSE'], np.nan)
        example_scpot_data.values = np.full_like(example_scpot_data, np.nan)

        # IDK IF I NEED THESE LINES, BUT ILL LEAVE THEM INSTEAD OF MESSING WITH IT
        # Get the corotation electric field
        E_cor = dm.corotation_electric_field(example_mec_data['R_sc'])

        # Get the spacecraft electric field
        E_sc = dm.E_convection(-example_mec_data['V_sc'], example_fgm_data['B_GSE'][:, 0:3])

        E_con = example_edi_data['E_GSE'] - E_cor - E_sc

        # Calculate the plasma convection field
        E_DIS = dm.E_convection(example_dis_data['V_DIS'],
                                example_fgm_data['B_GSE'][:, 0:3].reindex_like(example_dis_data['V_DIS']))
        E_DES = dm.E_convection(example_des_data['V_DES'],
                                example_fgm_data['B_GSE'][:, 0:3].reindex_like(example_des_data['V_DES']))

        return xr.Dataset({'E_EDI': example_edi_data['E_GSE'],
                           'B_GSE': example_fgm_data['B_GSE'],
                           'E_cor': E_cor,
                           'E_sc': E_sc,
                           'E_con': E_con,
                           'E_DIS': E_DIS,
                           'E_DES': E_DES,
                           'E_EDP': example_edp_data['E_GSE'],
                           'Kp': kp_data,
                           'Dst': dst_data,
                           'Sym-H': omni_data['Sym-H'],
                           'AE': omni_data['AE'],
                           'AL': omni_data['AL'],
                           'AU': omni_data['AU'],
                           'IEF': omni_data['IEF'],
                           'Scpot': example_scpot_data,
                           'R_sc': example_mec_data['R_sc'],
                           'V_sc': example_mec_data['V_sc'],
                           'L': example_mec_data['L'],
                           'MLT': example_mec_data['MLT']})