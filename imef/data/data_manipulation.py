import numpy as np
import xarray as xr
import datetime as dt
from warnings import warn
import scipy.optimize as optimize
from scipy.stats import binned_statistic, binned_statistic_2d, binned_statistic_dd
import torch

# For debugging purposes
np.set_printoptions(threshold=np.inf)

R_E = 6378.1  # km


def binned_avg_ds(ds, t_out):
    def ds_bin_avg(data_to_avg):
        avg, bin_edges, binnum = binned_statistic(t_in, data_to_avg,
                                                  statistic='mean',
                                                  bins=t_bins)
        return avg

    vars_out = {}

    t0 = dt.datetime.now()
    print('start time subtraction')

    # scipy does not like datetime64 so time has to be converted to floats
    #   - Subtracting lots of datetime64 can take a long time! (???)
    t_ref = np.min([ds['time'][0].data.astype('datetime64[D]'),
                    t_out[0].astype('datetime64[D]')])
    t_bins = (t_out - t_ref).astype('float')
    t_in = (ds['time'].data - t_ref).astype('float')

    t1 = dt.datetime.now()
    print('Elapsed time: subtraction: {0:0.4f}'.format((t1 - t0).total_seconds()))

    # Step through each variable in the dataset
    for name, var in ds.data_vars.items():
        # A time series
        if var.ndim == 1:
            if np.issubdtype(var.dtype, np.timedelta64):
                temp = ds_bin_avg(var.data.astype(float))
                temp = temp.astype(np.timedelta64)
            else:
                temp = ds_bin_avg(var.data)

            temp = xr.DataArray(temp,
                                dims=('time',),
                                coords={'time': t_out[:-1]},
                                name=name)

        # A vector of time series
        elif var.ndim == 2:
            temp_vars = []

            # Bin each component of the vector
            for idx in range(var.shape[1]):
                temp_vars.append(ds_bin_avg(var.data[:, idx]))

            # Combine bin-averaged components into an array
            temp = xr.DataArray(np.column_stack(temp_vars),
                                dims=('time', var.dims[1]),
                                coords={'time': t_out[:-1],
                                        var.dims[1]: var.coords[var.dims[1]]},
                                name=name)

        else:
            warn('Variable {0} has > 2 dimensions. Skipping'
                 .format(name))

        # Save the variables in the output dictionary
        vars_out[name] = temp

    t2 = dt.datetime.now()
    print('Elapsed time: binning:     {0:0.4f}'.format((t2 - t1).total_seconds()))

    # Create a new dataset
    return xr.Dataset(vars_out)


def binned_avg(data, t_out):
    '''
    Resample data by averaging into temporal bins

    Parameters
    ----------
    time : `xarray.DataArray` of `numpy.datetime64`
        Original time stamps of the data
    data : `numpy.array`
        Data to be averaged
    t_out : `numpy.datetime64`
        Leading edge of bins into which data should be averaged

    Returns
    -------
    avg : `xarray.DataArray`
        Resampled data
    '''

    # scipy does not like datetime64 so time has to be converted to floats
    t_ref = np.min([data['time'][0].data.astype('datetime64[D]'),
                    t_out[0].astype('datetime64[D]')])
    t_bins = (t_out - t_ref).astype('float')
    t_in = (data['time'].data - t_ref).astype('float')

    avg, bin_edges, binnumber = binned_statistic(t_in, data,
                                                 statistic='mean',
                                                 bins=t_bins)

    # The final time in t_out is the right-most bin edge of the final bin. Remove it.
    avg = xr.DataArray(avg, dims='time', coords={'time': t_out[:-1]})
    return avg


def bin_kp_r_theta(data, varname, r_range=(0, 10), dr=1, MLT_range=(0, 24), dMLT=1,
                   kp_bins=np.array([0, 0.67, 1.67, 2.67, 3.67, 4.67, 5.67, 6.67, 9.67])):
    '''
    Bin a variable in two dimensions, radial distance (r) and geomagnetic
    index (Kp).

    Parameters
    ----------
    data : `xarray.Dataset`
        Dataset containing the data to be binned. Should contain `varname`,
        'R_sc' (km) as the location at which each point was taken, and 'kp' as
        the geomagnetic index recorded at each point.
    varname : str
        Name of the variable in `data` to be binned and averaged. NaN values
        are ignored.
    r_range : (2) tuple, default=(0, 10)
        Radial range (R_E) over which to bin the data.
    dr : int, default=1
        Size of each radial bin (R_E)
    MLT_range : (2) tuple, default=(0, 24)
        MLT range (hrs) over which to keep the data: [min, max)
    dMLT : int
        Bin size in MLT (hrs)
    kp_bins : list of floats
        Bin edges of the Kp bins. Default bins are
        [0, 0.67, 1.67, 2.67, 3.67, 4.67, 5.67, 6.67, 9.67] corresponding to
        [0, 1-], [1, 2-], [2, 3-], [4, 5-], [5, 6-], [7, 10-]

    Returns
    -------
    counts : (L, N, M), int
        Number of points in each (kp, r) bin
    avg : (L, N, M), float
        Average value of `varname` in each bin
    kp_bins : (L), float
        Kp bin edges
    r_bins : (M), float
        Radial bin edges
    theta_bins : (N)
    '''

    # Create the radial and azimuthal bins
    r_bins = np.arange(r_range[0], r_range[1] + dr, dr)
    theta_bins = np.arange(MLT_range[0], MLT_range[1] + dMLT, dMLT) * 2 * np.pi / 24

    # Projection of the radial vector onto the xy-plane (ecliptic if GSE)
    r_mms = np.sqrt(data['R_sc'][:, 0] ** 2 + data['R_sc'][:, 1] ** 2) / R_E
    theta_mms = np.arctan2(data['R_sc'][:, 1], data['R_sc'][:, 0])
    theta_mms = (2 * np.pi + theta_mms) % 2 * np.pi
    sample = np.column_stack([data['kp'], r_mms, theta_mms])

    # Find all NaN values to remove
    igood = ~np.isnan(data[varname][:, 0].data)

    # Loop over each component
    bsr = None
    counts = []
    mean = []
    for idx in range(len(data[data[varname].dims[1]])):
        # counts
        bsr = binned_statistic_dd(sample[igood, :],
                                  values=data[varname][igood, idx].data,
                                  statistic='count',
                                  bins=[kp_bins, r_bins, theta_bins],
                                  binned_statistic_result=bsr)

        # Average value
        avg, bin_edges, num = (
            binned_statistic_dd(sample[igood, :],
                                values=data[varname][igood, idx].data,
                                statistic='mean',
                                binned_statistic_result=bsr)
        )

        # Store the results for each component
        counts.append(bsr[0])
        mean.append(avg)

    ds = xr.Dataset({varname + '_mean': (('kp', 'r', 'theta', 'comp'), np.stack(mean, axis=3)),
                     varname + '_counts': (('kp', 'r', 'theta', 'comp'), np.stack(counts, axis=3))},
                    coords={'kp': kp_bins[:-1],
                            'r': r_bins[:-1] + dr / 2,
                            'theta': theta_bins[:-1] + dMLT * 2 * np.pi / 24 / 2,
                            'comp': data[data[varname].dims[1]].data})

    return ds


def bin_r_kp(data, varname, r_range=(0, 10), dr=1, MLT_range=None,
             kp_bins=[0, 0.67, 1.67, 2.67, 3.67, 4.67, 5.67, 6.67, 9.67]):
    '''
    Bin a variable in two dimensions, radial distance (r) and geomagnetic
    index (Kp).

    Parameters
    ----------
    data : `xarray.Dataset`
        Dataset containing the data to be binned. Should contain `varname`,
        'R_sc' (km) as the location at which each point was taken, and 'kp' as
        the geomagnetic index recorded at each point.
    varname : str
        Name of the variable in `data` to be binned and averaged. NaN values
        are ignored.
    r_range : (2) tuple, default=(0, 10)
        Radial range (R_E) over which to bin the data.
    dr : int, default=1
        Size of each radial bin (R_E)
    kp_bins : list of floats
        Bin edges of the Kp bins. Default bins are
        [0, 0.67, 1.67, 2.67, 3.67, 4.67, 5.67, 6.67, 9.67] corresponding to
        [0, 1-], [1, 2-], [2, 3-], [4, 5-], [5, 6-], [7, 10-]
    MLT_range : (2) tuple, default=(0, 24)
        MLT range over which to keep the data: [min, max)

    Returns
    -------
    counts : (N, M), int
        Number of points in each (kp, r) bin
    avg : (N, M), float
        Average value of `varname` in each bin
    kp_bins : (N), float
        Kp bin edges
    r_bins : (M), float
        Radial bin edges
    '''

    # Reduce data in MLT
    if MLT_range is not None:
        # Convert MLT to radians
        theta_min = MLT_range[0] * 2 * np.pi / 24
        theta_max = MLT_range[1] * 2 * np.pi / 24

        # Determine azimuthal location of spacecraft
        theta_mms = np.atan2(data['R_sc'][:, 1], data['R_sc'][:, 0])

        # Remove all data outside of the given range
        data = data.sel(dict(time=((theta_mms >= theta_min)
                                   and (theta_mms < theta_max))))

    # Create the radial bins
    r_bins = np.arange(r_range[0], r_range[1] + dr, dr)

    # Projection of the radial vector onto the xy-plane (ecliptic if GSE)
    r_mms = np.sqrt(data['R_sc'][:, 0] ** 2 + data['R_sc'][:, 1] ** 2) / R_E

    # Find all NaN values to remove
    igood = ~np.isnan(data[varname][:, 1].data)

    # Counts
    counts, kp_bin_edge, r_bin_edge, num = binned_statistic_2d(x=data['kp'][igood].data,
                                                               y=r_mms[igood].data,
                                                               values=data[varname][igood, 1].data,
                                                               statistic='count',
                                                               bins=[kp_bins, r_bins])

    # Average value
    avg, kp_bin_edge, r_bin_edge, num = binned_statistic_2d(x=data['kp'][igood].data,
                                                            y=r_mms[igood].data,
                                                            values=data[varname][igood, 1].data,
                                                            statistic='mean',
                                                            bins=[kp_bins, r_bins])

    return counts, avg, kp_bins, r_bins


def cart2cyl(r_cart):
    r = np.sqrt(r_cart.loc[:, 'x'] ** 2 + r_cart.loc[:, 'y'] ** 2)
    phi = np.arctan2(r_cart.loc[:, 'y'], r_cart.loc[:, 'x'])
    z = r_cart.loc[:, 'z']

    # Combine into a vector
    #   - Convert r from an ndarray to a DataArray
    r_cyl = (xr.concat([xr.DataArray(r,
                                     dims='time',
                                     coords={'time': r_cart['time']}),
                        phi, z],
                       dim='cyl')
             .T.drop('cart')
             .assign_coords({'cyl': ['r', 'phi', 'z']})
             )

    return r_cyl


def cart2sphr(r_cart):
    # Position in spherical GSE coordinates
    r = np.linalg.norm(r_cart, ord=2, axis=1)
    phi = np.arctan2(r_cart[:, 1], r_cart[:, 0])
    theta = np.arccos(r_cart[:, 2] / r)

    # Combine into a vector
    #   - Convert r from an ndarray to a DataArray
    r_sphr = (xr.concat([xr.DataArray(r,
                                      dims='time',
                                      coords={'time': r_cart['time']}),
                         phi, theta],
                        dim='sphr')
              .T.drop('R_sc_index')
              .assign_coords({'sphr': ['r', 'phi', 'theta']})
              )

    return r_sphr


def corotation_electric_field(r_cart):
    # Position in spherical GSE coordinates
    r_cyl = cart2cyl(r_cart)

    # Corotation Electric Field
    #  - Radial component in the equatorial plane
    E_cor_polar = E_corot(r_cyl)

    # Matrix to transforma vector for cylindrical to cartesian coordinates
    xcyl2cart = xform_cyl2cart(r_cyl)

    # Transform the corotation electric field to cartesian coordinates
    E_cor = xcyl2cart.dot(E_cor_polar, dims='cyl')

    return E_cor


def corotating_frame(ds):
    '''
    Transform the electric field into a frame corotating with Earth.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Dataset with the electric field measurements to be transformed.
        Expects the variables:
            E_cor - Electric field due to corotation with respect to Earth
            E_sc - Electric field due to spacecraft motion (alt. typo E_con)
            E_* - Any electric field that needs to be transformed

    Returns
    -------
    ds : `xarray.Dataset`
        A new dataset with added variables that include
            E_*_inert - Electric field in the inertial frame
            E_*_corot - Electric field in the corotating frame
    '''

    new_vars = {}

    for name, var in ds.items():
        # Skip non-electric field variables
        if (not name.startswith('E_')) | (name.endswith(('_cor', '_con', '_sc'))):
            continue

        # Electric field in the inertial frame
        #   - Early files named the spacecraft E-field E_con(vection)
        try:
            new_vars[name + '_inert'] = var - ds['E_con']
        except KeyError:
            new_vars[name + '_inert'] = var - ds['E_sc']

        # Electric field in the corotating frame
        new_vars[name + '_corot'] = new_vars[name + '_inert'] - ds['E_cor']

    return ds.merge(new_vars)


def cyl2cart(r_cyl):
    x = r_cyl.loc[:, 'r'] * np.cos(r_cyl.loc[:, 'phi'])
    y = r_cyl.loc[:, 'r'] * np.sin(r_cyl.loc[:, 'phi'])
    z = r_cyl.loc[:, 'z'].drop('cyl')

    # Combine into a vector
    #   - Convert r from an ndarray to a DataArray
    r_cart = (xr.concat([x, y, z], dim='cyl')
              .T.rename({'cyl': 'cart'})
              .assign_coords({'time': r_cyl['time'],
                              'cart': ['x', 'y', 'z']})
              )

    return r_cart


def E_convection(v, b):
    '''
    Compute the convective electric field.

    Parameters
    ----------
    v : `xarray.DataArray`
        Velocity vector (km)
    b : `xarray.DataArray`
        Magnetic field vector (nT)

    Returns
    -------
    E : `xarray.DataArray`
        Convective electric field (mV/m)
    '''
    return xr.DataArray(-1e-3 * np.cross(v, b),
                        dims=('time', 'component'),
                        coords={'time': v['time'],
                                'component': ['x', 'y', 'z']})


def E_corot(r):
    # E_corot = C_corot*R_E/r^2 * 𝜙_hat
    # C_corot is found here
    omega_E = 2 * np.pi / (24 * 3600)  # angular velocity of Earth (rad/sec)
    B_0 = 3.12e4  # Earth mean field at surface (nT)
    R_E = 6371.2  # Earth radius (km)
    C_corot = omega_E * B_0 * R_E ** 2 * 1e-3  # V (nT -> T 1e-9, km**2 -> (m 1e3)**2)

    # Corotation Electric Field
    #  - Azimuthal component in the equatorial plane
    E_cor = (-C_corot * R_E / r.loc[:, 'r'] ** 2)
    E_cor = np.stack([E_cor,
                      np.zeros(len(E_cor)),
                      np.zeros(len(E_cor))], axis=1)
    E_cor = xr.DataArray(E_cor,
                         dims=['time', 'cyl'],
                         coords={'time': r['time'],
                                 'cyl': ['r', 'phi', 'z']})

    return E_cor


def expand_times(da, t_out):
    # Locate the data times within the packet times by using the packet times
    # as bin edges
    #   - scipy does not like datetime64 so time has to be converted to floats
    #   - Subtracting lots of datetime64 can take a long time! (???)
    t0 = np.min([da['time'][0].data.astype('datetime64[D]'),
                 t_out[0].astype('datetime64[D]')])
    dt_out = (t_out - t0).astype('float')
    dt_in = (da['time'].data - t0).astype('float')

    # Bin the data. The key here will be the bin number
    cnt, bin_edges, binnum = binned_statistic(dt_out, dt_out,
                                              statistic='count',
                                              bins=dt_in)

    # Test bins
    #   - Bin 0 is for data before the first packet time
    #   - Bin len(t_out) is for all data past the last packet time. There
    #     should be `sample_rate` points after the last packet time. If
    #     there are more, there is a problem: TODO - test
    if binnum[0] == 0:
        raise ValueError('Data times start before output times.')

    # Expand the input data
    result = da[binnum - 1].assign_coords({'time': t_out})

    return result


def generate_time_stamps(t_start, t_stop, t_res=np.timedelta64(5, 's')):
    '''
    Create an array of times spanning a time interval with a specified resolution.

    Parameters
    ----------
    t_start, t_stop : `numpy.datetime64`
        Start and end of the time interval, given as the begin times of the first
        and last samples
    t_res : `numpy.timedelta64`
        Sample interval

    Returns
    -------
    t_stamps : `numpy.datetime64`
        Timestamps spanning the given time interval and with the given resolution. Note
        that the last point in the array is the end time of the last sample. This is to
        work better with `scipy.binned_statistic`.
    '''
    # Find the start time
    t_ref = t_start.astype('datetime64[m]').astype('datetime64[ns]')
    t_start = t_start - ((t_start - t_ref) % t_res)

    # Find the end time -- it should be after the final time
    t_ref = t_stop.astype('datetime64[m]').astype('datetime64[ns]')
    dt_round = (t_stop - t_ref) % t_res
    t_stop += (t_res - dt_round)

    # Time at five second intervals.
    #  - We want t_end to be included in the array as the right-most edge of the time interval
    t_stamps = np.arange(t_start, t_stop + t_res, t_res)

    return t_stamps


def interp(ds, t_out, method='linear', extrapolate=False):
    if extrapolate:
        kwargs = {'fill_value': 'extrapolate'}
    else:
        kwargs = None

    return ds.interp(time=t_out, method=method, kwargs=kwargs)


def interp_gaps_ds(ds, t_out, extrapolate=False):
    for name, var in ds.data_vars.items():
        pass


def interp_over_gaps(data, t_out, extrapolate=False):
    '''
    Interpolate data being careful to avoid interpolating over data gaps

    Parameters
    ----------
    t_out : `numpy.datetime64`
        Times to which the data should be interpolated
    data : `xarray.DataArray`
        Data to be interpolated with coordinate 'time'
    t_delta : `numpy.timedelta64`
        Sampling interval of the data

    Returns
    -------
    result : `xarray.DataArray`
        Data interpolated to the given timestamps via nearest neighbor averaging
    '''
    if extrapolate:
        kwargs = {'fill_value': 'extrapolate'}
    else:
        kwargs = None

    # Data gaps are any time interval larger than a sampling interval
    N_dt = np.round(data['time'].diff(dim='time') / data['dt_plus']).astype(int)

    # Find the last sample in each contiguous segment
    #   - The last data point in the array is the last sample in the last contiguous segment
    idx_gaps = np.argwhere(N_dt.data > 1)[:, 0]
    idx_gaps = np.append(idx_gaps, len(data['time']))

    # Number of data gaps
    N_gaps = len(idx_gaps)
    if (N_gaps - 1) > 0:
        warn('{0} data gaps found in dataset'
             .format(N_gaps - 1))

    # Interpolate each contiguous segment
    temp = []
    istart = 0
    for igap in idx_gaps:
        temp.append(data[dict(time=slice(istart, igap + 1))]
                    .interp({'time': t_out}, method='nearest', kwargs=kwargs)
                    .dropna(dim='time')
                    )
        istart = igap + 1

    # Combine each contiguous segment into a single array
    return xr.concat(temp, dim='time')


def sphr2cyl(r_sphr):
    # Radial position in equatorial plane
    #  - for some reason phi_cyl = np.arcsin(y/r_cyl) did not work
    r = r_sphr.loc[:, 'r'] * np.sin(r_sphr.loc[:, 'theta'])
    phi = r_sphr.loc[:, 'phi'].drop('sphr')
    z = r_sphr.loc[:, 'r'] * np.cos(r_sphr.loc[:, 'theta'])

    r_cyl = (xr.concat([r, phi, z], dim='cyl').T
             .assign_coords({'cyl': ['r', 'phi', 'z']})
             )

    return r_cyl


def spacecraft_electric_field(b, dt=np.timedelta64(5, 's')):
    # Move FGM time stamps to the beginning of the sample interval
    fgm_data = fgm_data.assign_coords(
        {'begin_time': fgm_data['time'] - (1e6 * fgm_data['time_delta']).astype('datetime64[us]')})


def V_corot(r):
    # Corotation Electric Potential
    #  - Azimuthal component in the equatorial plane
    V_cor = -92100 * R_E / r.loc[:, 'r']

    return V_cor


def xform_cyl2cart(r):
    # Unit vectors
    x_hat = np.stack([np.cos(r.loc[:, 'phi']),
                      -np.sin(r.loc[:, 'phi']),
                      np.zeros(len(r))], axis=1)
    y_hat = np.stack([np.sin(r.loc[:, 'phi']),
                      np.cos(r.loc[:, 'phi']),
                      np.zeros(len(r))], axis=1)
    z_hat = np.repeat(np.array([[0, 0, 1]]), len(r), axis=0)

    xcyl2cart = xr.DataArray(np.stack([x_hat, y_hat, z_hat], axis=2),
                             dims=('time', 'cyl', 'cart'),
                             coords={'time': r['time'],
                                     'cyl': ['r', 'phi', 'z'],
                                     'cart': ['x', 'y', 'z']})

    return xcyl2cart.transpose('time', 'cart', 'cyl')


def xform_cart2cyl(r):
    # Unit vectors
    phi = np.arctan2(r.loc[:, 'y'], r.loc[:, 'x'])
    r_hat = np.stack([np.cos(phi), np.sin(phi), np.zeros(len(r))], axis=1)
    phi_hat = np.stack([-np.sin(phi), np.cos(phi), np.zeros(len(r))], axis=1)
    z_hat = np.repeat(np.array([[0, 0, 1]]), len(r), axis=0)

    xcart2cyl = xr.DataArray(np.stack([r_hat, phi_hat, z_hat], axis=2),
                             dims=('time', 'cart', 'cyl'),
                             coords={'time': r['time'],
                                     'cart': ['x', 'y', 'z'],
                                     'cyl': ['r', 'phi', 'z']})

    return xcart2cyl.transpose('time', 'cyl', 'cart')


def aaaaaaaaaaaaaaaaaaaa():
    pass


def cart2polar(pos_cart, factor=1, shift=0):
    '''
    Rotate cartesian position coordinates to polar coordinates
    '''
    r = (np.sqrt(np.sum(pos_cart[:, [0, 1]] ** 2, axis=1)) * factor)
    phi = (np.arctan2(pos_cart[:, 1], pos_cart[:, 0])) + shift
    pos_polar = xr.concat([r, phi], dim='polar').T.assign_coords({'polar': ['r', 'phi']})

    return pos_polar


def rot2polar(vec, pos, dim):
    '''
    Rotate vector from cartesian coordinates to polar coordinates
    '''
    # Polar unit vectors
    phi = pos.loc[:, 'phi']
    r_hat = xr.concat([np.cos(phi).expand_dims(dim), np.sin(phi).expand_dims(dim)], dim=dim)
    phi_hat = xr.concat([-np.sin(phi).expand_dims(dim), np.cos(phi).expand_dims(dim)], dim=dim)

    # Rotate vector to polar coordinates
    Vr = vec[:, [0, 1]].dot(r_hat, dims=dim)
    Vphi = vec[:, [0, 1]].dot(phi_hat, dims=dim)
    v_polar = xr.concat([Vr, Vphi], dim='polar').T.assign_coords({'polar': ['r', 'phi']})

    return v_polar


def remove_spacecraft_efield(edi_data, fgm_data, mec_data):
    # The assumption is that this is done prior to remove_corot_efield
    # This should be done BEFORE converting edi_data to polar

    # E = v x B, 1e-3 converts units to mV/m
    E_sc = 1e-3 * np.cross(mec_data['V_sc'][:, :3], fgm_data['B_GSE'][:, :3])

    # Make into a DataArray to subtract the data easier
    E_sc = xr.DataArray(E_sc,
                        dims=['time', 'E_index'],
                        coords={'time': edi_data['time'],
                                'E_index': ['Ex', 'Ey', 'Ez']},
                        name='E_sc')

    original_egse = edi_data['E_GSE']

    # Remove E_sc from the measured electric field
    edi_data['E_GSE'] = edi_data['E_GSE'] - E_sc

    edi_data['Original_E_GSE'] = original_egse

    edi_data['E_sc'] = E_sc

    return edi_data


def remove_corot_efield(edi_data, mec_data):
    # This should be done after remove_spacecraft_efield
    # This should be done BEFORE converting edi_data to polar

    # E_corot = C_corot*R_E/r^2 * 𝜙_hat
    # C_corot is found here
    omega_E = 2 * np.pi / (24 * 3600)  # angular velocity of Earth (rad/sec)
    B_0 = 3.12e4  # Earth mean field at surface (nT)
    R_E = 6371.2  # Earth radius (km)
    C_corot = omega_E * B_0 * R_E ** 2 * 1e-3  # V (nT -> T 1e-9, km**2 -> (m 1e3)**2)

    # Position in spherical GSE coordinates
    r_sphr = np.linalg.norm(mec_data['R_sc'], ord=2,
                            axis=mec_data['R_sc'].get_axis_num('R_sc_index'))
    phi_sphr = np.arctan2(mec_data['R_sc'][:, 1], mec_data['R_sc'][:, 0])
    theta_sphr = np.arccos(mec_data['R_sc'][:, 2] / r_sphr)

    # Radial position in equatorial plane
    #  - for some reason phi_cyl = np.arcsin(y/r_cyl) did not work
    r_cyl = r_sphr * np.sin(theta_sphr)
    phi_cyl = np.arctan2(mec_data['R_sc'][:, 1], mec_data['R_sc'][:, 0])
    z_cyl = mec_data['R_sc'][:, 2]

    # Corotation Electric Field
    #  - taken from data_manipulation.remove_corot_efield
    #  - Azimuthal component in the equatorial plane (GSE)
    E_corot = (-92100 * R_E / r_cyl ** 2)
    E_corot = np.stack([np.zeros(len(E_corot)), E_corot, np.zeros(len(E_corot))], axis=1)
    E_corot = xr.DataArray(E_corot,
                           dims=['time', 'E_index'],
                           coords={'time': mec_data['time'],
                                   'E_index': ['r', 'phi', 'z']})

    # Unit vectors
    x_hat = np.stack([np.cos(phi_cyl), -np.sin(phi_cyl), np.zeros(len(phi_cyl))], axis=1)
    y_hat = np.stack([np.sin(phi_cyl), np.cos(phi_cyl), np.zeros(len(phi_cyl))], axis=1)

    # Transform to Cartesian
    Ex_corot = np.einsum('ij,ij->i', E_corot, x_hat)
    Ey_corot = np.einsum('ij,ij->i', E_corot, y_hat)
    Ez_corot = np.zeros(len(x_hat))
    E_gse_corot = xr.DataArray(np.stack([Ex_corot, Ey_corot, Ez_corot], axis=1),
                               dims=['time', 'E_index'],
                               coords={'time': E_corot['time'],
                                       'E_index': ['x', 'y', 'z']})

    minus_spacecraft_egse = edi_data['E_GSE'].copy(deep=True)

    # For some reason edi_data['E_GSE'] = edi_data['E_GSE'] - E_gse_corot results in all nan's. strange. This works tho
    edi_data['E_GSE'][:, 0] = edi_data['E_GSE'][:, 0] - E_gse_corot[:, 0]
    edi_data['E_GSE'][:, 1] = edi_data['E_GSE'][:, 1] - E_gse_corot[:, 1]
    edi_data['E_GSE'][:, 2] = edi_data['E_GSE'][:, 2] - E_gse_corot[:, 2]

    edi_data['no_spacecraft_E_GSE'] = minus_spacecraft_egse

    edi_data['E_Corot'] = E_gse_corot

    return edi_data


def slice_data_by_time(full_data, ti, te):
    # Here is where the desired time and data values will be placed
    time = np.array([])
    wanted_value = np.array([])

    # Slice the wanted data and put them into 2 lists
    for counter in range(0, len(full_data)):
        # The data at each index is all in one line, separated by whitespace. Separate them
        new = str.split(full_data.iloc[counter][0])

        # Create the time at that point
        time_str = str(new[0]) + '-' + str(new[1]) + '-' + str(new[2]) + 'T' + str(new[3][:2]) + ':00:00'

        # Make a datetime object out of time_str
        insert_time_beg = dt.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')

        # We know for Kp that the middle of the bin is 1.5 hours past the beginning of the bin (which is insert_time_beg)
        insert_time_mid = insert_time_beg + dt.timedelta(hours=1.5)

        # If the data point is within the time range that is desired, insert the time and the associated kp index
        # The other datasets use the dates as datetime64 objects. So must use those instead of regular datetime objects
        if insert_time_mid + dt.timedelta(hours=1.5) > ti and insert_time_mid - dt.timedelta(hours=1.5) < te:
            insert_kp = new[7]
            time = np.append(time, [insert_time_mid])
            wanted_value = np.append(wanted_value, [insert_kp])

    return time, wanted_value


def slice_dst_data(full_data, ti, te):
    start_of_first_month = dt.datetime(year=ti.year, month=ti.month, day=1)
    if te.month != 12:
        end_of_last_month = dt.datetime(year=te.year, month=te.month + 1, day=1) - dt.timedelta(hours=1) + dt.timedelta(
            microseconds=1)
    else:
        end_of_last_month = dt.datetime(year=te.year + 1, month=1, day=1) - dt.timedelta(hours=1) + dt.timedelta(
            microseconds=1)
    time_list = datetime_range(start_of_first_month, end_of_last_month, dt.timedelta(hours=1))

    for counter in range(0, len(full_data)):
        one_day = str.split(full_data.iloc[counter][0])
        one_day.pop(0)

        # There are some days that have large dips in dst, eclipsing -100 nT. when this happens, the whitespace between numbers is gone, and looks something like -92-105-111-119-124-109.
        # this if statement removes those strings, splits them up, and inserts them back into the one_day list as separated strings
        if len(one_day) != 24:
            for counter in range(0, 24):
                number = one_day[counter]
                if len(number) > 4:
                    too_big_number = str.split(number, '-')
                    if too_big_number[0] == '':
                        too_big_number.pop(0)
                    too_big_number = (np.array(too_big_number).astype('int64') * -1).astype('str').tolist()
                    one_day[counter:counter] = too_big_number
                    one_day.pop(counter + len(too_big_number))

        if counter == 0:
            dst_list = np.array(one_day)
        else:
            dst_list = np.append(dst_list, [one_day])

    counter2 = 0
    while time_list[counter2] < ti:
        counter2 += 1

    counter3 = len(time_list) - 1
    while time_list[counter3] > te:
        counter3 -= 1

    time_list = time_list[counter2:counter3]
    dst_list = dst_list[counter2:counter3]

    return time_list, dst_list


def datetime_range(start, end, delta):
    datetimes = []
    current = start
    while current < end:
        datetimes.append(current)
        current += delta
    return datetimes


def expand_5min_kp(time, kp):
    # This is for a very specific case, where you are given times at the very beginning of a day, and you only want 5 minute intervals. All other cases should be run through expand_kp

    # Expanding the kp values is easy, as np.repeat does this for us.
    # Eg: np.repeat([1,2,3],3) = [1,1,1,2,2,2,3,3,3]
    new_kp = np.repeat(kp, 36)

    new_times = np.array([])

    # Iterate through every time that is given
    for a_time in time:
        # Put the first time value into the xarray. This corresponds to the start of the kp window
        new_times = np.append(new_times, [a_time - dt.timedelta(hours=1.5)])

        # There are 36 5-minute intervals in the 3 hour window Kp covers. We have the time in the middle of the 3 hour window.
        # Here all the values are created (other than the start of the window, done above) by incrementally getting 5 minutes apart on both sides of the middle value. This is done 18 times
        # However we must make an exception when the counter is 0, because otherwise it will put the middle of the window twice
        for counter in range(18):
            if counter == 0:
                new_time = a_time
                new_times = np.append(new_times, [new_time])
            else:
                new_time_plus = a_time + counter * dt.timedelta(minutes=5)
                new_time_minus = a_time - counter * dt.timedelta(minutes=5)
                new_times = np.append(new_times, [new_time_plus, new_time_minus])

    # The datetime objects we want are created, but out of order. Put them in order
    new_times = np.sort(new_times, axis=0)

    return new_times, new_kp


def expand_kp(kp_times, kp, time_to_expand_to):
    # Note that this function is capable of doing the same thing as expand_5min, and more.

    # Also note that this function can be used for other indices and such as long as they are inputted in the same format as the Kp data is

    # Because Datetime objects that are placed into xarrays get transformed into datetime64 objects, and the conventional methods of changing them back do not seem to work,
    # You have to make a datetime64 version of the kp_times so that they can be subtracted correctly

    # Iterate through all times and convert them to datetime64 objects
    for time in kp_times:
        # The timedelta is done because the min function used later chooses the lower value in the case of a tie. We want the upper value to be chosen
        if type(time) == type(dt.datetime(2015, 9, 10)):
            time64 = np.datetime64(time - dt.timedelta(microseconds=1))
        elif type(time) == type(np.datetime64(1, 'Y')):
            time64 = time - np.timedelta64(1,
                                           'ms')  # This thing only works for 1 millisecond, not 1 microsecond. Very sad
        else:
            raise TypeError('Time array must contain either datetime or datetime64 objects')

        if time == kp_times[0]:
            datetime64_kp_times = np.array([time64])
        else:
            datetime64_kp_times = np.append(datetime64_kp_times, [time64])

    # This will be used to find the date closest to each time given
    # It will find the value closest to each given value. In other words it is used to find the closest time to each time from the given list
    absolute_difference_function = lambda list_value: abs(list_value - given_value)

    # Iterate through all times that we want to expand kp to
    for given_value in time_to_expand_to:

        # Find the closest value
        closest_value = min(datetime64_kp_times, key=absolute_difference_function)

        # Find the corresponding index of the closest value
        index = np.where(datetime64_kp_times == closest_value)

        if given_value == time_to_expand_to[0]:
            # If this is the first iteration, create an ndarray containing the kp value at the corresponding index
            new_kp = np.array([kp[index]])
        else:
            # If this is not the first iteration, combine the new kp value with the existing ones
            new_kp = np.append(new_kp, kp[index])

    return new_kp


def interpolate_data_like(data, data_like):
    data = data.interp_like(data_like)

    return data


def create_timestamps(data, ti, te):
    '''
    Convert times to UNIX timestamps.
    Parameters
    ----------
    data : `xarray.DataArray`
        Data with coordinate 'time' containing the time stamps (np.datetime64)
    ti, te : `datetime.datetime`
        Time interval of the data
    Returns
    -------
    ti, te : float
        Time interval converted to UNIX timestamps
    timestamps :
        Time stamps converted to UNIX times
    '''

    # Define the epoch and one second in np.datetime 64. This is so we can convert np.datetime64 objects to timestamp values
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')

    # Round up the end time by one microsecond so the bins aren't marginally incorrect.
    # this is good enough for now, but this is a line to keep an eye on, as it will cause some strange things to happen due to the daylight savings fix later on.
    # if off by 1 microsecond, will erroneously gain/lose 1 hour
    if (te.second != 0) and te.second != 30:
        te = te + dt.timedelta(microseconds=1)

    ti_datetime = ti
    te_datetime = te

    # Convert the start and end times to a unix timestamp
    # This section adapts for the 4 hour time difference (in seconds) that timestamp() automatically applies to datetime. Otherwise the times from data and these time will not line up right
    # This appears because timestamp() corrects for local time difference, while the np.datetime64 method did not
    # This could be reversed and added to all the times in data, but I chose this way.
    # Note that this can cause shifts in hours if the timezone changes
    ti = ti.timestamp() - 14400
    te = te.timestamp() - 14400

    # This is to account for daylight savings
    # lazy fix: check to see if ti-te is the right amount of time. If yes, move on. If no, fix by changing te to what it should be
    # This forces the input time to be specifically 1 day of data, otherwise this number doesn't work.
    # Though maybe the 86400 could be generalized using te-ti before converting to timestamp. Possible change there
    # Though UTC is definitely something to be done, time permitting (i guess it is in UTC. need to figure out at some point)
    # This will only work for exactly 1 day of data being downloaded. It will be fine for sample and store_edi data,
    # however if running a big download that goes through a daylight savings day, there will be an issue

    if ti_datetime + dt.timedelta(days=1) == te_datetime:
        if te - ti > 86400:
            te -= 3600
        elif te - ti < 86400:
            te += 3600

    # Create the array where the unix timestamp values go
    # The timestamp values are needed so we can bin the values with binned_statistic
    # Note that the data argument must be an xarray object with a 'time' dimension so that this works. Could be generalized at some point
    timestamps = (data['time'].values - unix_epoch) / one_second

    return ti, te, timestamps


def get_5min_times(data, vars_to_bin, timestamps, ti, te):
    # Get the times here. This way we don't have to rerun getting the times for every single variable that is being binned
    number_of_bins = (te - ti) / 300
    count, bin_edges, binnum = binned_statistic(x=timestamps, values=data[vars_to_bin[0]], statistic='count',
                                                bins=number_of_bins, range=(ti, te))

    # Create an nparray where the new 5 minute interval datetime64 objects will go
    new_times = np.array([], dtype=object)

    # Create the datetime objects and add them to new_times
    for time in bin_edges:
        # Don't run if the time value is the last index in bin_edges. There is 1 more bin edge than there is mean values
        # This is because bin_edges includes an extra edge to encompass all the means
        # As a result, the last bin edge (when shifted to be in the middle of the dataset) doesn't correspond to a mean value
        # So it must be ignored so that the data will fit into a new dataset
        if time != bin_edges[-1]:
            # Convert timestamp to datetime object
            new_time = dt.datetime.utcfromtimestamp(time)

            # Add 2.5 minutes to place the time in the middle of each bin, rather than the beginning
            new_time = new_time + dt.timedelta(minutes=2.5)

            # Add the object to the nparray
            new_times = np.append(new_times, [new_time])

    # Return the datetime objects of the 5 minute intervals created in binned_statistic
    return new_times


def bin_5min(data, vars_to_bin, index_names, ti, te):
    '''
    Bin one day's worth of data into 5-minute intervals.
    Parameters
    ----------
    data
    vars_to_bin
    index_names
    ti, te
    Returns
    -------
    complete_data
    '''
    # Any variables that are not in var_to_bin are lost (As they can't be mapped to the new times otherwise)
    # Note that it is possible that NaN values appear in the final xarray object. This is because there were no data points in those bins
    # To remove these values, use xarray_object = test.where(np.isnan(test['variable_name']) == False, drop=True) (Variable has no indices)
    # Or xarray_object = xarray_object.where(np.isnan(test['variable_name'][:,0]) == False, drop=True) (With indices)

    # Also note that in order for this to work correctly, te-ti must be a multiple of 5 minutes.
    # This is addressed in the get_xxx_data functions, since they just extend the downloaded times by an extra couple minutes or whatever

    # In order to bin the values properly, we need to convert the datetime objects to integers. I chose to use unix timestamps to do so
    ti, te, timestamps = create_timestamps(data, ti, te)
    new_times = get_5min_times(data, vars_to_bin, timestamps, ti, te)

    number_of_bins = (te - ti) / 300

    # Iterate through every variable (and associated index) in the given list
    for var_counter in range(len(vars_to_bin)):
        if index_names[var_counter] == '':
            # Since there is no index associated with this variable, there is only 1 thing to be meaned. So take the mean of the desired variable
            means, bin_edges_again, binnum = binned_statistic(x=timestamps, values=data[vars_to_bin[var_counter]],
                                                              statistic='mean', bins=number_of_bins, range=(ti, te))
            std, bin_edges_again, binnum = binned_statistic(x=timestamps, values=data[vars_to_bin[var_counter]],
                                                            statistic='std', bins=number_of_bins, range=(ti, te))

            # Create the dataset for the meaned variable
            new_data = xr.Dataset(coords={'time': new_times})

            # Fix the array so it will fit into the dataset
            var_values = means.T
            var_values_std = std.T

            # Put the data into the dataset
            new_data[vars_to_bin[var_counter]] = xr.DataArray(var_values, dims=['time'], coords={'time': new_times})
            new_data[vars_to_bin[var_counter] + '_std'] = xr.DataArray(var_values_std, dims=['time'],
                                                                       coords={'time': new_times})
        else:
            # Empty array where the mean of the desired variable will go
            means = np.array([[]])
            stds = np.array([[]])

            # Iterate through every variable in the associated index
            for counter in range(len(data[index_names[var_counter] + '_index'])):
                # Find the mean of var_to_bin
                # mean is the mean in each bin, bin_edges is the edges of each bin in timestamp values, and binnum is which values go in which bin
                mean, bin_edges_again, binnum = binned_statistic(x=timestamps,
                                                                 values=data[vars_to_bin[var_counter]][:, counter],
                                                                 statistic='mean', bins=number_of_bins, range=(ti, te))
                std, bin_edges_again, binnum = binned_statistic(x=timestamps,
                                                                values=data[vars_to_bin[var_counter]][:, counter],
                                                                statistic='std', bins=number_of_bins, range=(ti, te))

                # If there are no means yet, designate the solved mean value as the array where all of the means will be stored. Otherwise combine with existing data
                if means[0].size == 0:
                    means = [mean]
                    stds = [std]
                else:
                    means = np.append(means, [mean], axis=0)
                    stds = np.append(stds, [std], axis=0)

            # Create the new dataset where the 5 minute bins will go
            new_data = xr.Dataset(coords={'time': new_times, index_names[var_counter] + '_index': data[
                index_names[var_counter] + '_index']})

            # Format the mean values together so that they will fit into new_data
            var_values = means.T
            var_values_std = stds.T

            # Put in var_values
            new_data[vars_to_bin[var_counter]] = xr.DataArray(var_values,
                                                              dims=['time', index_names[var_counter] + '_index'],
                                                              coords={'time': new_times})
            new_data[vars_to_bin[var_counter] + '_std'] = xr.DataArray(var_values_std, dims=['time', index_names[
                var_counter] + '_index'], coords={'time': new_times})

        # If this is the first run, designate the created data as the dataset that will hold all the data. Otherwise combine with the existing data
        if var_counter == 0:
            complete_data = new_data
        else:
            complete_data = xr.merge([complete_data, new_data])

    return complete_data


def get_A(min_Lvalue, max_Lvalue):
    # E=AΦ
    # A=-⛛ (-1*Gradient)
    # In polar coordinates, A=(1/△r, 1/r△Θ), where r is the radius and Θ is the azimuthal angle
    # In this case, r=L, △r=△Θ=1

    # First we need to make the gradient operator. Since we have electric field data in bins labeled as 4.5, 5.5, ... and want potential values at integer values 1,2,...
    # We use the central difference operator to get potential values at integer values.

    # For example, E_r(L=6.5, MLT=12.5)=Φ(7,13)+Φ(7,12) - Φ(6,13)+Φ(6,12)
    # Recall matrix multiplication rules to know how we can reverse engineer each row in A knowing the above
    # So for E_r, 1 should be where Φ(7,13)+Φ(7,12) is, and -1 should be where Φ(6,13)+Φ(6,12) is
    # For the example above, that row in A looks like [0.....-1,-1, 0....1, 1, 0...0].

    # For E_az, things are slightly different, E_az(L=6.5, MLT=12.5) = 1/L * 24/2π * [Φ(7,13)+Φ(6,13)]/2 - [Φ(7,12)+Φ(6,12)]/2
    # 1/L represents the 1/r△Θ in the gradient operator, and 24/2π is the conversion from radians to MLT
    # All of the rows follow the E_r and E_az examples, and as a result A has 4 values in each row

    # This runs assuming the E vector is organized like the following:
    # E=[E_r(L=0,MLT=0), E_az(0,0), E_r(0,1), E_az(0,1)...E_r(1,0), E_az(1,0)....]
    # This may be changed later, especially if a 3rd dimension is added

    # The edge case where MLT=23.5 must be treated separately, because it has to use MLT=23 and MLT=0 as its boundaries

    L_range = int(max_Lvalue - min_Lvalue + 1)
    A = np.zeros((2 * 24 * L_range, 24 * (L_range + 1)))

    # In order to index it nicely, we must subtract the minimum value from the max value, so we can start indexing at 0
    # As a result, L_counter does not always represent the actual L value
    # In this case, the real L value is calculated by adding L_counter by min_Lvalue
    matrix_value_r = 1
    for L_counter in range(L_range):
        # This only accounts for MLT values from 0.5 to 22.5. The value where MLT = 23.5 is an exception handled at the end
        for MLT_counter in range(0, 23):
            # Here is where we implement the A values that give E_r
            A[get_A_row(L_counter, MLT_counter), get_A_col(L_counter, MLT_counter)] = -matrix_value_r
            A[get_A_row(L_counter, MLT_counter), get_A_col(L_counter, MLT_counter, 1)] = -matrix_value_r
            A[get_A_row(L_counter, MLT_counter), get_A_col(L_counter, MLT_counter, 24)] = matrix_value_r
            A[get_A_row(L_counter, MLT_counter), get_A_col(L_counter, MLT_counter, 25)] = matrix_value_r

            # Here is where we implement the A values that give E_az at the same point that the above E_r was found
            matrix_value_az = 1 * 24 / (2 * np.pi) / (L_counter + min_Lvalue)
            A[get_A_row(L_counter, MLT_counter, 1), get_A_col(L_counter, MLT_counter)] = -matrix_value_az
            A[get_A_row(L_counter, MLT_counter, 1), get_A_col(L_counter, MLT_counter, 24)] = -matrix_value_az
            A[get_A_row(L_counter, MLT_counter, 1), get_A_col(L_counter, MLT_counter, 1)] = matrix_value_az
            A[get_A_row(L_counter, MLT_counter, 1), get_A_col(L_counter, MLT_counter, 25)] = matrix_value_az

        # Where MLT=23.5 is implemented
        # E_r
        A[get_A_row(L_counter, other=46), get_A_col(L_counter, other=23)] = -matrix_value_r
        A[get_A_row(L_counter, other=46), get_A_col(L_counter)] = -matrix_value_r
        A[get_A_row(L_counter, other=46), get_A_col(L_counter, other=47)] = matrix_value_r
        A[get_A_row(L_counter, other=46), get_A_col(L_counter, other=24)] = matrix_value_r

        # E_az
        matrix_value_az = 1 * 24 / (2 * np.pi) / (L_counter + min_Lvalue)
        A[get_A_row(L_counter, other=47), get_A_col(L_counter, other=23)] = -matrix_value_az
        A[get_A_row(L_counter, other=47), get_A_col(L_counter, other=47)] = -matrix_value_az
        A[get_A_row(L_counter, other=47), get_A_col(L_counter)] = matrix_value_az
        A[get_A_row(L_counter, other=47), get_A_col(L_counter, other=24)] = matrix_value_az

    # Conversion factor between kV/Re and mV/m
    # The -1 comes from E=-⛛V. A=-⛛, therefore we need the -1 in front
    constant = -1 / 6.3712
    A *= constant

    return A


def get_A_row(L, MLT=0, other=0):
    return 48 * L + 2 * MLT + other


def get_A_col(L, MLT=0, other=0):
    return 24 * L + MLT + other


def get_C(min_Lvalue, max_Lvalue):
    # C is the hessian, or the second derivative matrix. It is used to smooth the E=AΦ relation when solving the inverse problem
    # The overall procedure used to find A is used again here (reverse engineering the values of A), however there are more values per row
    # Also, the central difference operator is not used here, so there are no halving of values
    # For the example y=Cx: y(L=6, MLT=12) = x(L=5, 12 MLT)+x(L=7, 12 MLT)+x(L=6, 11 MLT)+x(L=6, 13 MLT)-4*x(L=6, 12 MLT)

    # Like A, the edge cases must be accounted for. While the MLT edge cases can be handled the same way as in A, there are now edge cases in L.
    # The L edge cases are handled by **ignoring the lower values apparently**
    # For example, if L=4 was the lowest L value measured, then y=x(L=5, 0 MLT)+x(L=4, 23 MLT)+x(L=4, 1 MLT)-4*x(L=4, 0 MLT)

    # But, because C is a square matrix, we can use a different, much easier method to create this matrix than we did with A
    # From the above example, we know that every value down the diagonal is -4. So we can use np.diag(np.ones(dimension)) to make a square matrix with ones across the diagonal and 0 elsewhere
    # Multiplying that by -4 gives us all the -4 values we want.
    # We can use the same method to create a line of ones one above the diagonal by using np.diag(np.ones(dimension-1), 1)
    # The above method can be refactored to create a line of ones across any diagonal of the matrix
    # So we just create a couple lines of ones and add them all together to create the C matrix

    L_range = int(max_Lvalue - min_Lvalue + 1)
    MLT = 24

    # For the example y(L=6, MLT=12), this creates the -4*x(L=6, MLT=12)
    minusfour = -4 * np.diag(np.ones(MLT * (L_range + 1)))

    # These create the x(L=6, MLT=13) and x(L=6, MLT=11) respectively
    MLT_ones = np.diag(np.ones(MLT * (L_range + 1) - 1), 1)
    moreMLT_ones = np.diag(np.ones(MLT * (L_range + 1) - 1), -1)

    # These create the x(L=7, MLT=12) and x(L=5, MLT=12) respectively
    L_ones = np.diag(np.ones(MLT * (L_range + 1) - MLT), MLT)
    moreL_ones = np.diag(np.ones(MLT * (L_range + 1) - MLT), -MLT)

    # Add the ones matrices and create C
    C = minusfour + MLT_ones + moreMLT_ones + L_ones + moreL_ones

    # Nicely, this method handles the edge L cases for us, so we don't have to worry about those.
    # However we do need to handle the edge MLT cases, since both MLT=0 and MLT=23 are incorrect as is

    # This loop fixes all the edge cases except for the very first and very last row in C, as they are fixed differently than the rest
    for counter in range(1, L_range + 1):
        # Fixes MLT=23
        C[MLT * counter - 1][MLT * counter] = 0
        C[MLT * counter - 1][MLT * (counter - 1)] = 1

        # Fixes MLT=0 at the L value 1 higher than the above statement
        C[MLT * counter][MLT * counter - 1] = 0
        C[MLT * counter][MLT * (counter + 1) - 1] = 1

    # Fixes the first row
    C[0][MLT - 1] = 1
    # Fixes the last row
    C[MLT * (L_range + 1) - 1][MLT * L_range] = 1

    return C


def calculate_potential(imef_data, name_of_variable):
    # Determine the L range that the data uses
    min_Lvalue = imef_data['L'][0, 0].values
    max_Lvalue = imef_data['L'][-1, 0].values

    # Find the number of bins relative to L and MLT
    # nL is the number of L values in E, not Φ. So there will be nL+1 in places. There are 6 L values in E, but 7 in Φ (As L is taken at values of 4.5, 5.5, etc in E, but 4, 5, etc in Φ)
    nL = int(max_Lvalue - min_Lvalue + 1)
    nMLT = 24

    # Get the electric field data and make them into vectors. MUST BE POLAR COORDINATES
    E_r = imef_data[name_of_variable][:, :, 0].values.flatten()
    E_az = imef_data[name_of_variable][:, :, 1].values.flatten()

    # Create the number of elements that the potential will have
    nElements = 24 * nL
    E = np.zeros(2 * nElements)

    # Reformat E_r and E_az so that they are combined into 1 vector following the format
    # [E_r(L=4, MLT=0), E_az(L=4, MLT=0), E_r(L=4, MLT=1), E_az(L=4, MLT=1), ... E_r(L=5, MLT=0), E_az(L=5, MLT=0), ...]
    for index in range(0, nElements):
        E[2 * index] = E_r[index]
        E[2 * index + 1] = E_az[index]

    # Create the A matrix
    A = get_A(min_Lvalue, max_Lvalue)

    # Create the C matrix
    C = get_C(min_Lvalue, max_Lvalue)

    # Define the tradeoff parameter γ
    gamma = 2.51e-4

    # Solve the inverse problem according to the equation in Matsui 2004 and Korth 2002
    # V=(A^T * A + γ * C^T * C)^-1 * A^T * E
    V = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A) + gamma * np.dot(C.T, C)), A.T), E)
    V = V.reshape(nL + 1, nMLT)

    return V


def calculate_potential_2(imef_data, name_of_variable, guess):
    # Determine the L range that the data uses
    min_Lvalue = imef_data['L'][0, 0].values
    max_Lvalue = imef_data['L'][-1, 0].values

    # Find the number of bins relative to L and MLT
    # nL is the number of L values in E, not Φ. So there will be nL+1 in places. There are 6 L values in E, but 7 in Φ (As L is taken at values of 4.5, 5.5, etc in E, but 4, 5, etc in Φ)
    nL = int(max_Lvalue - min_Lvalue + 1)
    nMLT = 24

    # Get the electric field data and make them into vectors. MUST BE POLAR COORDINATES
    E_r = imef_data[name_of_variable][:, :, 0].values.flatten()
    E_az = imef_data[name_of_variable][:, :, 1].values.flatten()

    # Create the number of elements that the potential will have
    nElements = 24 * int(max_Lvalue - min_Lvalue + 1)
    E = np.zeros(2 * nElements)

    # Reformat E_r and E_az so that they are combined into 1 vector following the format
    # [E_r(L=4, MLT=0), E_az(L=4, MLT=0), E_r(L=4, MLT=1), E_az(L=4, MLT=1), ... E_r(L=5, MLT=0), E_az(L=5, MLT=0), ...]
    for index in range(0, nElements):
        E[2 * index] = E_r[index]
        E[2 * index + 1] = E_az[index]

    # Create the A matrix
    A = get_A(min_Lvalue, max_Lvalue)

    # Create the C matrix
    C = get_C(min_Lvalue, max_Lvalue)

    # Define the tradeoff parameter γ
    gamma = 2.51e-4

    # HERE IS THE DIFFERENCES FROM CALCULATE_POTENTIAL

    def loss(v, A, E, C, gamma):
        function = np.dot(np.transpose(np.dot(A, v) - E), (np.dot(A, v) - E)) + gamma * np.dot(
            np.dot(np.transpose(v), C), v)
        return function

    def grad_loss(v, A, E, C, gamma):
        return 2 * np.transpose(A) @ A @ v - 2 * np.transpose(A) @ E + 2 * gamma * np.transpose(C) @ C @ v

    # Solve the inverse problem according to the equation in Matsui 2004 and Korth 2002
    V = optimize.minimize(loss, guess, args=(A, E, C, gamma), method="CG", jac=grad_loss)
    optimized = V.x
    # V = V.reshape(nL + 1, nMLT)

    return V

def get_NN_inputs(imef_data, remove_nan=True, get_target_data=True, use_values=['Kp'], usetorch=True):
    # This could be made way more efficient if I were to make the function not download all the data even if it isn't used. But for sake of understandability (which this has little of anyways)

    if 'Kp' in use_values:
        Kp_data = imef_data['Kp']
    if 'Dst' in use_values:
        Dst_data = imef_data['DST']
    if 'Symh' in use_values:
        Symh_data = imef_data['SYMH']

    if remove_nan == True:
        imef_data = imef_data.where(np.isnan(imef_data['E_EDI'][:, 0]) == False, drop=True)

    # Note that the first bits of data cannot be used, because the first 60 times dont have Kp values and whatnot. Will become negligible when done on a large amount of data
    for counter in range(60, len(imef_data['time'].values)):
        new_data_line = []
        if 'Kp' in use_values:
            Kp_index_data = Kp_data.values[counter - 60:counter].tolist()
            new_data_line += Kp_index_data
        if 'Dst' in use_values:
            Dst_index_data = Dst_data.values[counter - 60:counter].tolist()
            new_data_line += Dst_index_data
        if 'Symh' in use_values:
            Symh_index_data = Symh_data.values[counter - 60:counter].tolist()
            new_data_line += Symh_index_data

        # Along with the indices, we include 3 extra values to train on: The distance from the Earth (L), cos(MLT), and sin(MLT)
        # There are two lines here, one for the nanoseconds timedelta version of the files, one for the normal MLT versions of the files. Only the MLT version should be used in the future, but this line remains just in case
        # For nanosecond timedeltas
        # the_rest_of_the_data = np.array([imef_data['L'].values[counter], np.cos(np.pi / 12 * imef_data['MLT'].values[counter] / np.timedelta64(1, 'h')), np.sin(np.pi / 12 * imef_data['MLT'].values[counter] / np.timedelta64(1, 'h'))]).tolist()
        # For MLT
        the_rest_of_the_data = np.array([imef_data['L'].values[counter], np.cos(np.pi / 12 * imef_data['MLT'].values[counter]), np.sin(np.pi / 12 * imef_data['MLT'].values[counter])]).tolist()
        new_data_line += the_rest_of_the_data

        if counter == 60:
            design_matrix_array = [new_data_line]
        else:
            design_matrix_array.append(new_data_line)

    if usetorch==True:
        design_matrix_array = torch.tensor(design_matrix_array)
    else:
        design_matrix_array = np.array(design_matrix_array)

    if get_target_data == True:
        # The convection electric field is the total electric field (E_EDI), minus the spacecraft electric field (E_con), minus the corotation electric field (E_cor)
        efield_data = imef_data['E_EDI'].values[60:, :] - imef_data['E_con'].values[60:, :] - imef_data['E_cor'].values[60:, :]

        if usetorch==True:
            efield_data = torch.from_numpy(efield_data)

        return design_matrix_array, efield_data
    else:
        return design_matrix_array