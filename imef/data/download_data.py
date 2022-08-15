import os
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import shutil
from urllib import request
from contextlib import closing
import requests

from heliopy.data import omni
import imef.data.data_manipulation as dm
from pymms.data import edi, util, fgm, fpi, edp
from pymms.sdc import mrmms_sdc_api as mms_api

# Kp and DSCOVR data
import util as dd_util

np.set_printoptions(threshold=np.inf)


def get_data(sc, instr, mode, level, t0, t1, dt_out=None):
    if dt_out is not None:
        if dt_out < dt.timedelta(seconds=5):
            raise ValueError('dt_out must be >= 5s.')
        extrapolate = False
        repeat = False

    # EDI
    if instr == 'edi':
        data = get_edi_data(sc, mode, level, t0, t1)

        # Rename the time delta variable
        dt_plus_vname = '_'.join((sc, 'edi', 't', 'delta', 'plus', mode, level))
        dt_minus_vname = '_'.join((sc, 'edi', 't', 'delta', 'minus', mode, level))
        data = data.rename({dt_plus_vname: 'dt_plus',
                            dt_minus_vname: 'dt_minus'})

        # Set the time delta variable
        data['dt_plus'] = np.timedelta64(np.int(5e9), 'ns')
        data['dt_minus'] = np.timedelta64(np.int(0), 'ns')

    # FGM
    elif instr == 'fgm':
        data = get_fgm_data(sc, mode, t0, t1)

        # Move FGM time stamps to the beginning of the sample interval
        dt_fgm = (1e6 * data['time_delta']).astype('timedelta64[us]')
        data = data.drop('time_delta')
        data = data.assign_coords({'dt_plus': 2 * dt_fgm,
                                   'dt_minus': 0 * dt_fgm})

        # Time has to be assigned after dt_plus and dt_minus because those
        # are time dependent and must have their times be reset, too
        data = data.assign_coords({'time': data['time'] - dt_fgm})

    # MEC
    elif instr == 'mec':
        data = get_mec_data(sc, mode, level, t0, t1)

        data = (data.drop(['R_sc_index', 'V_sc_index'])
                .assign_coords({'cart': ['x', 'y', 'z']})
                .rename({'R_sc_index': 'cart',
                         'V_sc_index': 'cart'})
                )

        # Parse the time resolution from the attributes (e.g., '30 s')
        dt_mec = data.attrs['Time_resolution'].split(' ')
        dt_mec = np.timedelta64(np.int(dt_mec[0]), dt_mec[1])

        # Add time resolution
        data = data.assign_coords({'dt_plus': dt_mec,
                                   'dt_minus': np.timedelta64(0, 's')})

        # If resampling, extrapolating is OK
        extrapolate = True

    # DIS
    elif instr == 'dis':
        data = get_dis_data(sc, mode, level, t0, t1)

        data = data.rename({'Epoch_minus_var': 'dt_minus',
                            'Epoch_plus_var': 'dt_plus'})
        data.assign_coords({'time': data['time'] - data['dt_minus']})
        data['dt_plus'] += data['dt_minus']
        data['dt_minus'] = np.timedelta64(0, 's')

    # DES
    elif instr == 'des':
        data = get_des_data(sc, mode, level, t0, t1)

        data = data.rename({'Epoch_minus_var': 'dt_minus',
                            'Epoch_plus_var': 'dt_plus'})
        data.assign_coords({'time': data['time'] - data['dt_minus']})
        data['dt_plus'] += data['dt_minus']
        data['dt_minus'] = np.timedelta64(0, 's')

    # EDP
    elif instr == 'edp':
        edp_data_slow = edp.load_data(sc, 'slow', level, start_date=t0, end_date=t1)
        edp_data_fast = edp.load_data(sc, 'fast', level, start_date=t0, end_date=t1)

        # Rename the indices so that the fast and slow data can be combined
        e_slow_labl_vname = '_'.join((sc, instr, 'label1', 'slow', level))
        e_fast_labl_vname = '_'.join((sc, instr, 'label1', 'fast', level))
        edp_data_slow = edp_data_slow.rename({e_slow_labl_vname: 'E_index'})
        edp_data_fast = edp_data_fast.rename({e_fast_labl_vname: 'E_index'})

        # Expand the time delta to have the same length as time
        dt_slow_vname = '_'.join((sc, instr, 'deltap', 'slow', level))
        dt_slow = (np.repeat(edp_data_slow[dt_slow_vname]
                             .rename({dt_slow_vname: 'time'}),
                             len(edp_data_slow['time']))
                   .assign_coords({'time': edp_data_slow['time']})
                   )

        # Expand the time delta to have the same length as time
        dt_fast_vname = '_'.join((sc, instr, 'deltap', 'fast', level))
        dt_fast = (np.repeat(edp_data_fast[dt_fast_vname]
                             .rename({dt_fast_vname: 'time'}),
                             len(edp_data_fast['time']))
                   .assign_coords({'time': edp_data_fast['time']})
                   )

        # Combine slow and fast variables
        e_edp = xr.concat([edp_data_slow['E_GSE'], edp_data_fast['E_GSE']],
                          dim='time').sortby('time')
        dt_edp = (xr.concat([dt_slow, dt_fast], dim='time')
                  .sortby('time')
                  .astype('timedelta64['
                          + edp_data_slow[dt_slow_vname].attrs['units']
                          + ']')
                  )

        # Combine the data into a dataset
        data = xr.Dataset({'E_GSE': e_edp,
                           'dt_plus': dt_edp,
                           'dt_minus': dt_edp})

        # Adjust the time stamps
        #   - dt_minus needs to be assigned because both
        #     dt_plus and dt_minus are views of dt_edp
        data = data.assign_coords({'time': data['time'] - data['dt_minus']})
        data['dt_plus'] *= 2
        data = data.assign({'dt_minus': (('time',), np.repeat(np.timedelta64(0, 's'), len(dt_edp)))})

    # SCPOT
    elif instr == 'scpot':
        data = get_scpot_data(sc, level, t0, t1)

    else:
        raise ValueError('"{0}" is not a valid instrument. Choose from '
                         '(edi, fgm, mec, dis, des, edp, scpot)')

    # Resample in time
    data = resample(data, t0, t1, dt_out,
                    extrapolate=extrapolate, repeat=repeat)

    return data


def get_orbit_number(sc, t0, t1, dt_out=None):
    '''
    Get the orbit numbers associated with a given time range.

    Parameters
    ----------
    sc : str
        Spacecraft identifier
    t0, t1 : `datetime.datetime`
        Start and end time of the data interval
    dt_out : `datetime.timedelta`
        Sample interval if the data is to be resampled

    Returns
    -------
    orbit_data : `xarray.DataArray`
        Orbit numbers
    '''

    # Get the orbit information
    #   - Time interval must encompass entire orbit
    #   - Put some padding on either side (orbits have been <~ 3 days [2022])
    orbits = mms_api.mission_events('orbit',
                                    t0 - dt.timedelta(days=5),
                                    t1 + dt.timedelta(days=5),
                                    sc=sc)

    # Orbit interval relative to time stamp
    dt_plus = [o1 - o0 for o0, o1 in zip(orbits['tstart'], orbits['tend'])]
    dt_minus = np.repeat(np.timedelta64(0, 's'), len(orbits['start_orbit']))

    # Orbit number
    orb_num = xr.DataArray(orbits['start_orbit'],
                           dims=('time',),
                           coords={'time': orbits['tstart'],
                                   'dt_plus': ('time', dt_plus),
                                   'dt_minus': ('time', dt_minus)})

    # Resample
    if dt_out is not None:
        orbit_data = resample(orb_num, t0, t1, dt_out, repeat=True)

    return orbit_data


def get_kp_data_v2(t0, t1, dt_out=None):
    data = []

    # Time intervals
    kp_util = dd_util.Kp_Downloader()
    intervals = kp_util.intervals(t0, t1)

    # Load the data
    for interval in intervals:
        data.append(kp_util.load_file(interval))

    # Combine the data into a single dataset
    kp_data = xr.concat(data, dim='time').sortby('time')

    # Resample
    if dt_out is not None:
        kp_data = resample(kp_data['Kp'], t0, t1, dt_out, repeat=True)

    return kp_data


def get_scpot_data(sc, level, t0, t1):
    scpot_slow = edp.load_scpot(sc=sc, mode='slow', level=level,
                                start_date=t0, end_date=t1)
    scpot_fast = edp.load_scpot(sc=sc, mode='fast', level=level,
                                start_date=t0, end_date=t1)

    # Expand the time delta to have the same length as time
    dt_slow_vname = '_'.join((sc, 'edp', 'deltap', 'slow', level))
    dt_slow = (np.repeat(scpot_slow[dt_slow_vname]
                         .rename({dt_slow_vname: 'time'}),
                         len(scpot_slow['time']))
               .assign_coords({'time': scpot_slow['time']})
               )

    # Expand the time delta to have the same length as time
    dt_fast_vname = '_'.join((sc, 'edp', 'deltap', 'fast', level))
    dt_fast = (np.repeat(scpot_fast[dt_fast_vname]
                         .rename({dt_fast_vname: 'time'}),
                         len(scpot_fast['time']))
               .assign_coords({'time': scpot_fast['time']})
               )

    # Combine slow and fast variables
    Vsc = (xr.concat([scpot_slow['Vsc'], scpot_fast['Vsc']], dim='time')
           .sortby('time')
           )
    dt_scpot = (xr.concat([dt_slow, dt_fast], dim='time')
                .sortby('time')
                .astype('timedelta64['
                        + scpot_slow[dt_slow_vname].attrs['units']
                        + ']')
                )

    # Adjust the time stamps
    #   - dt_minus needs to be assigned because both
    #     dt_plus and dt_minus are views of dt_edp
    Vsc = Vsc.assign_coords({'time': Vsc['time'] - dt_scpot,
                             'dt_plus': ('time', 2 * dt_scpot),
                             'dt_minus': ('time', 0 * dt_scpot)})

    return Vsc


def get_dscovr_data():
    raise NotImplementedError


def get_dst_data(t0, t1, dt_out=None):
    data = []

    # Time intervals
    dst_util = dd_util.Dst_Downloader()
    intervals = dst_util.intervals(t0, t1)

    # Load the data
    for interval in intervals:
        data.append(dst_util.load_file(interval))

    # Combine the data into a single dataset
    dst_data = xr.concat(data, dim='time').sortby('time')

    # Resample
    if dt_out is not None:
        dst_data = resample(dst_data['Dst'], t0, t1, dt_out, repeat=True)

    return dst_data


def get_omni_data(t0, t1, dt_out=None):
    full_omni_data = omni.hro2_1min(t0 - dt.timedelta(microseconds=1), t1)

    full_omni_data = full_omni_data.to_dataframe()

    # The first value of every month appears twice (I assume it's a bug). Remove the duplicate value
    full_omni_data = full_omni_data.drop_duplicates()

    # Convert pandas dataframe to an xarray dataset
    full_omni_data = full_omni_data.to_xarray()

    omni_data = xr.Dataset(
        coords={'Time': full_omni_data['Time'], 'V_index': ['Vx', 'Vy', 'Vz'], 'B_index': ['Bx', 'By', 'Bz']})

    # Concatenate the V and B values so that they will fit inside the above dataset.
    V_values = np.vstack((full_omni_data['Vx'], full_omni_data['Vy'], full_omni_data['Vz'])).T
    B_values = np.vstack((full_omni_data['BX_GSE'], full_omni_data['BY_GSE'], full_omni_data['BZ_GSE'])).T

    # Put the values into the new dataset
    omni_data['V_OMNI'] = xr.DataArray(V_values, dims=['Time', 'V_index'], coords={'Time': omni_data['Time']})
    omni_data['B_OMNI'] = xr.DataArray(B_values, dims=['Time', 'B_index'], coords={'Time': omni_data['Time']})
    omni_data['Sym-H'] = xr.DataArray(full_omni_data['SYM_H'].values, dims=['Time'], coords={'Time': omni_data['Time']})
    omni_data['AE'] = xr.DataArray(full_omni_data['AE_INDEX'].values, dims=['Time'], coords={'Time': omni_data['Time']})
    omni_data['AL'] = xr.DataArray(full_omni_data['AL_INDEX'].values, dims=['Time'], coords={'Time': omni_data['Time']})
    omni_data['AU'] = xr.DataArray(full_omni_data['AU_INDEX'].values, dims=['Time'], coords={'Time': omni_data['Time']})
    omni_data = omni_data.assign_coords({'dt_plus': np.timedelta64(1, 'm'), 'dt_minus': np.timedelta64(0, 'h')})

    # Rename time so that it is the same as the other datasets, making concatenation easier
    omni_data = omni_data.rename({'Time': 'time'})

    # Put the data into the dataset
    omni_data['IEF'] = dm.calculate_IEF(omni_data)

    if dt_out is not None:
        V_omni = resample(omni_data['V_OMNI'], t0, t1, dt_out, repeat=True)
        B_omni = resample(omni_data['B_OMNI'], t0, t1, dt_out, repeat=True)
        Sym_h = resample(omni_data['Sym-H'], t0, t1, dt_out, repeat=True)
        AE = resample(omni_data['AE'], t0, t1, dt_out, repeat=True)
        AL = resample(omni_data['AL'], t0, t1, dt_out, repeat=True)
        AU = resample(omni_data['AU'], t0, t1, dt_out, repeat=True)
        IEF = resample(omni_data['IEF'], t0, t1, dt_out, repeat=True)
        omni_data = xr.Dataset(coords={'time': V_omni['time'], 'V_index': ['Vx', 'Vy', 'Vz'], 'B_index': ['Bx', 'By', 'Bz']})
        omni_data['V_OMNI'] = V_omni
        omni_data['B_OMNI'] = B_omni
        omni_data['Sym-H']= Sym_h
        omni_data['AE'] = AE
        omni_data['AL'] = AL
        omni_data['AU'] = AU
        omni_data['IEF'] = IEF

    return omni_data


def resample(data, t0, t1, dt_out, extrapolate=False, repeat=False):
    # Generate a set of timestamps
    t_out = dm.generate_time_stamps(np.datetime64(t0),
                                    np.datetime64(t1),
                                    np.timedelta64(dt_out))

    if (data['dt_plus'] == np.timedelta64(dt_out)).all():
        pass

    # Upsample
    #   - Less than two samples per target sampling interval
    elif (data['dt_plus'] > (0.5 * np.timedelta64(dt_out))).any():

        # Repeat values
        if repeat:
            data = dm.expand_times(data, t_out[:-1])

        # Interpolate
        else:
            data = dm.interp_over_gaps(data, t_out[:-1],
                                       extrapolate=extrapolate)

    # Downsample
    #   - Two or more samples per target sampling interval
    elif (data['dt_plus'] <= (0.5 * np.timedelta64(dt_out))).any():
        if isinstance(data, xr.DataArray):
            func = dm.binned_avg
        else:
            func = dm.binned_avg_ds

        # Simple average
        data = func(data, t_out)

    return data.assign_coords({'dt_plus': dt_out,
                               'dt_minus': np.timedelta64(0, 's')})


def download_ftp_files(remote_location, local_location, fname_list):
    '''
    Transfer files from FTP location to local location
    Parameters
    ----------
    remote_location : str
        Location on the FTP server where files are located
    local_location : str
        Local location where remote FTP files are to be copied
    fname_list : list of str
        List of files on the FTP site to be transferred
    '''
    # this always downloads the file, in contrast to the other which only downloads it if the file doesn't already exist in data/kp
    # for fname in fname_list:
    #     # note this will redownload a file that has already been downloaded.
    #     with open(local_location + fname, 'wb') as f:
    #         with closing(request.urlopen(remote_location + fname)) as r:
    #             shutil.copyfileobj(r, f)

    # this does the same thing, but if the file name already exists, it doesn't download the data. note that this doesn't work flawlessly,
    # as any file that is downloaded from an incomplete month/year will not be updated, which can cause problems. while just deleting the file would fix it, meh
    for fname in fname_list:
        # Check if they exist
        if os.path.isfile(local_location + fname) == 0:
            # If they do not exist, create the new file and copy down the data
            # Note that this will not update existing files. May want to figure that out at some point
            with open(local_location + fname, 'wb') as f:
                with closing(request.urlopen(remote_location + fname)) as r:
                    shutil.copyfileobj(r, f)


def download_html_data(remote_location_list, local_location_list):
    # apparently this line will break if the data/dst/ directory doesn't exist already. maybe find a way to fix
    for remote_location, local_location in zip(remote_location_list, local_location_list):
        # this is super finicky if used in the long term, but it will redownload all relatively recent files while not wasting time on the older ones
        if os.path.isfile(local_location) == 0 or int(local_location[13:17]) > 2021:
            r = requests.get(remote_location, allow_redirects=True)
            open(local_location, 'wb').write(r.content)


def read_txt_files(fname_list, local_location=None, mode='Kp'):
    '''
    Reads data into a Pandas dataframe
    Parameters
    ----------
    fname_list : list of str
        Files containing Kp index
    local_location : str
        Path to where files are stored, if it isn't included in fname_list
    Returns
    -------
    full_data : `pandas.DataFrame`
        Kp data
    '''
    if mode == 'Kp':
        header = 29
        footer = 0

    # Combine all of the needed files into one dataframe
    for fname in fname_list:
        if mode == 'Dst' and int(fname[13:17]) == 2021 and int(fname[18:20]) >= 8 or mode == 'Dst' and int(
                fname[13:17]) >= 2022:
            header = 34
            footer = 55
        elif mode == 'Dst' and int(fname[13:17]) >= 2020:
            header = 34
            footer = 40
        elif mode == 'Dst' and int(fname[13:17]) < 2020:
            header = 28
            footer = 41

        # Read file into a pandas dataframe, and remove the text at the top
        if local_location is not None:
            oneofthem = pd.read_table(local_location + fname, header=header, skipfooter=footer)
        else:
            oneofthem = pd.read_table(fname, header=header, skipfooter=footer)

        if fname == fname_list[0]:
            # If this is the first time going through the loop, designate the created dataframe as where all the data will go
            full_data = oneofthem
        else:
            # Otherwise, combine the new data with the existing data
            full_data = pd.concat([full_data, oneofthem], ignore_index=True)

    return full_data


def get_edi_data(sc, mode, level, ti, te, binned=False):
    '''
    Load EDI data. Time tags of EDI data are moved to the beginning of the
    accumulation interval to facilitate binning of other data products.
    Parameters
    ----------
    sc : str
        Spacecraft identifier: {'mms1', 'mms2', 'mms3', 'mms4'}
    mode : str
        Data rate mode: {'srvy', 'slow', 'fast', 'brst'}
    level : str
        Data level: {'l1a', 'l2'}
    ti, te: `datetime.datetime`
        Start and end of the data interval
    binned : bool
        Bin/average data into 5-minute intervals
    Returns
    -------
    edi_data : `xarray.Dataset`
        EDI electric field data
    '''
    # binned=True bins the data into 5 minute bins in the intervals (00:00:00, 00:05:00, 00:10:00, etc)
    # For example, the bin 00:10:00 takes all the data from 00:07:30 and 00:12:30 and bins them
    # The first bin will not have enough data to bin into 5 minute intervals (It goes into the previous day).
    # But we also don't want values to overlap from day to day, so we have to take away another 2.5 minutes from the end so that we don't see repeats
    if binned == True:
        ti = ti - dt.timedelta(minutes=2.5)
        te = te - dt.timedelta(minutes=2.5)

        # the bin_5min program requires a multiple of 5 minutes from start to end (so no data is left off)
        # check if the time range given is a multiple of 5 minutes, and if it isn't, add onto the end whatever time is needed to make it a 5 minute interval
        if ((te - ti) % dt.timedelta(minutes=5)) / dt.timedelta(seconds=1) != 0:
            te = te + dt.timedelta(
                minutes=(5 - ((te - ti) / dt.timedelta(minutes=5) - int((te - ti) / dt.timedelta(minutes=5))) * 5))

    tm_vname = '_'.join((sc, 'edi', 't', 'delta', 'minus', mode, level))

    # Get EDI data
    # edi_data = edi.load_data(sc, mode, level, optdesc='efield', start_date=ti, end_date=te)
    edi_data = edi.load_efield(sc=sc, mode=mode, level=level, start_date=ti, end_date=te)

    # Timestamps begin on 0's and 5's and span 5 seconds. The timestamp is at
    # the weighted mean of all beam hits. To get the beginning of the timestamp,
    # subtract the time's DELTA_MINUS. But this is inaccurate by a few nanoseconds
    # so we have to round to the nearest second.
    edi_time = edi_data['Epoch'] - edi_data[tm_vname].astype('timedelta64[ns]')
    edi_time = [(t - tdelta)
                if tdelta.astype(int) < 5e8
                else (t + np.timedelta64(1, 's') - tdelta)
                for t, tdelta in zip(edi_time.data, edi_time.data - edi_time.data.astype('datetime64[s]'))
                ]

    # Replace the old time data with the corrected time data
    edi_data['Epoch'] = edi_time

    # Rename Epoch to time for consistency across all data files
    edi_data = edi_data.rename({'Epoch': 'time'})

    # This should probably be in the other get_xxx_data functions as well, but they have not caused problems just yet (most of the time they are downloaded in conjunction with this anyways)
    if len(edi_data['time']) == 0:
        raise IndexError('No data in this time range')

    # We want to use E_GSE, so we bin through E_GSE.
    # Note that binning results in the rest of the variables (everything but E_GSE) being dropped.
    if binned == True:
        edi_data = dm.bin_5min(edi_data, ['E_GSE'], ['E'], ti, te)

    edi_data = edi_data.rename({'E_index': 'cart'})
    edi_data['cart'] = ['x', 'y', 'z']

    return edi_data


def get_fgm_data(sc, mode, ti, te, binned=False):
    '''
    Load FGM data.
    Parameters
    ----------
    sc : str
        Spacecraft identifier: {'mms1', 'mms2', 'mms3', 'mms4'}
    mode : str
        Data rate mode: {'srvy', 'slow', 'fast', 'brst'}
    level : str
        Data level: {'l1a', 'l2'}
    ti, te: `datetime.datetime`
        Start and end of the data interval
    binned : bool
        Bin/average data into 5-minute intervals
    Returns
    -------
    fgm_data : `xarray.Dataset`
        FGM magnetic field data
    '''
    # binned=True bins the data into 5 minute bins in the intervals (00:00:00, 00:05:00, 00:10:00, etc)
    # For example, the bin 00:10:00 takes all the data from 00:07:30 and 00:12:30 and bins them
    # The first bin will not have enough data to bin into 5 minute intervals (It goes into the previous day).
    # But we also don't want values to overlap from day to day, so we have to take away another 2.5 minutes from the end so that we don't see repeats
    if binned == True:
        ti = ti - dt.timedelta(minutes=2.5)
        te = te - dt.timedelta(minutes=2.5)

        # the bin_5min program requires a multiple of 5 minutes from start to end (so no data is left off)
        # check if the time range given is a multiple of 5 minutes, and if it isn't, add onto the end whatever time is needed to make it a 5 minute interval
        if ((te - ti) % dt.timedelta(minutes=5)) / dt.timedelta(seconds=1) != 0:
            te = te + dt.timedelta(
                minutes=(5 - ((te - ti) / dt.timedelta(minutes=5) - int((te - ti) / dt.timedelta(minutes=5))) * 5))

    # Get FGM data
    fgm_data = fgm.load_data(sc=sc, mode=mode, start_date=ti, end_date=te)

    if binned == True:
        fgm_data = dm.bin_5min(fgm_data, ['B_GSE'], ['b'], ti, te)

    return fgm_data


def get_mec_data(sc, mode, level, ti, te, binned=False):
    '''
    Load MEC data.
    Parameters
    ----------
    sc : str
        Spacecraft identifier: {'mms1', 'mms2', 'mms3', 'mms4'}
    mode : str
        Data rate mode: {'srvy', 'brst'}
    level : str
        Data level: {'l2'}
    ti, te : `datetime.datetime`
        Start and end of the data interval
    binned : bool
        Bin/average data into 5-minute intervals
    Returns
    -------
    mec_data : `xarray.Dataset`
        MEC ephemeris data
    '''
    # binned=True bins the data into 5 minute bins in the intervals (00:00:00, 00:05:00, 00:10:00, etc)
    # For example, the bin 00:10:00 takes all the data from 00:07:30 and 00:12:30 and bins them
    # The first bin will not have enough data to bin into 5 minute intervals (It goes into the previous day).
    # But we also don't want values to overlap from day to day, so we have to take away another 2.5 minutes from the end so that we don't see repeats
    if binned == True:
        ti = ti - dt.timedelta(minutes=2.5)
        te = te - dt.timedelta(minutes=2.5)

        # the bin_5min program requires a multiple of 5 minutes from start to end (so no data is left off)
        # check if the time range given is a multiple of 5 minutes, and if it isn't, add onto the end whatever time is needed to make it a 5 minute interval
        if ((te - ti) % dt.timedelta(minutes=5)) / dt.timedelta(seconds=1) != 0:
            te = te + dt.timedelta(
                minutes=(5 - ((te - ti) / dt.timedelta(minutes=5) - int((te - ti) / dt.timedelta(minutes=5))) * 5))

    # The names of the variables that will be downloaded
    r_vname = '_'.join((sc, 'mec', 'r', 'gse'))
    v_vname = '_'.join((sc, 'mec', 'v', 'gse'))
    mlt_vname = '_'.join((sc, 'mec', 'mlt'))
    l_dip_vname = '_'.join((sc, 'mec', 'l', 'dipole'))

    # The names of the indices used for the radius (?) and velocity data
    r_lbl_vname = '_'.join((sc, 'mec', 'r', 'gse', 'label'))
    v_lbl_vname = '_'.join((sc, 'mec', 'v', 'gse', 'label'))

    # Get MEC data
    mec_data = util.load_data(sc, 'mec', mode, level,
                              optdesc='epht89d', start_date=ti, end_date=te,
                              variables=[r_vname, v_vname, mlt_vname, l_dip_vname])

    # Rename variables
    mec_data = mec_data.rename({r_vname: 'R_sc',
                                r_lbl_vname: 'R_sc_index',
                                v_vname: 'V_sc',
                                v_lbl_vname: 'V_sc_index',
                                mlt_vname: 'MLT',
                                l_dip_vname: 'L',
                                'Epoch': 'time',
                                })

    if binned == True:
        mec_data = dm.bin_5min(mec_data, ['V_sc', 'R_sc', 'L', 'MLT'], ['V_sc', 'R_sc', '', ''], ti, te)

    return mec_data


def get_edp_data(sc, level, ti, te, binned=False):
    '''
    Load EDP data.
    Parameters
    ----------
    sc : str
        Spacecraft identifier: {'mms1', 'mms2', 'mms3', 'mms4'}
    level : str
        Data level: {'l2'}
    ti, te : `datetime.datetime`
        Start and end of the data interval
    binned : bool
        Bin/average data into 5-minute intervals
    Returns
    -------
    edp_data : `xarray.Dataset`
        EDP electric field data
    '''
    # binned=True bins the data into 5 minute bins in the intervals (00:00:00, 00:05:00, 00:10:00, etc)
    # For example, the bin 00:10:00 takes all the data from 00:07:30 and 00:12:30 and bins them
    # The first bin will not have enough data to bin into 5 minute intervals (It goes into the previous day).
    # But we also don't want values to overlap from day to day, so we have to take away another 2.5 minutes from the end so that we don't see repeats
    if binned == True:
        ti = ti - dt.timedelta(minutes=2.5)
        te = te - dt.timedelta(minutes=2.5)

        # the bin_5min program requires a multiple of 5 minutes from start to end (so no data is left off)
        # check if the time range given is a multiple of 5 minutes, and if it isn't, add onto the end whatever time is needed to make it a 5 minute interval
        if ((te - ti) % dt.timedelta(minutes=5)) / dt.timedelta(seconds=1) != 0:
            te = te + dt.timedelta(
                minutes=(5 - ((te - ti) / dt.timedelta(minutes=5) - int((te - ti) / dt.timedelta(minutes=5))) * 5))

    edp_data_fast = edp.load_data(sc, 'fast', level, start_date=ti, end_date=te)
    edp_data_slow = edp.load_data(sc, 'slow', level, start_date=ti, end_date=te)

    # the label for the E_index of fast efield data
    e_fast_vname_label = '_'.join((sc, 'edp', 'label1', 'fast', level))

    # the label for the E_index of slow efield data
    e_slow_vname_label = '_'.join((sc, 'edp', 'label1', 'slow', level))

    edp_data_fast = edp_data_fast.rename({'E_GSE': 'E_EDP',
                                          e_fast_vname_label: 'E_index'})

    edp_data_slow = edp_data_slow.rename({'E_GSE': 'E_EDP',
                                          e_slow_vname_label: 'E_index'})

    # So should I be merging as EDP_FAST and EDP_SLOW separately or no? I'm not sure
    # Right now it is being combined into 1 variable, but that causes an error on certain days (eg. 1/18/16)
    # Combine fast and slow data
    edp_data = xr.merge([edp_data_fast, edp_data_slow])

    # Reorganize the data to sort by time
    edp_data = edp_data.sortby('time')

    # Cast the float32 values to float64 values to prevent overflow when binning
    edp_data['E_EDP'].values = edp_data['E_EDP'].values.astype('float64')

    if binned == True:
        edp_data = dm.bin_5min(edp_data, ['E_EDP'], ['E'], ti, te)

    return edp_data


def get_omni_data_old(ti, te):
    # Download the omni data as a timeseries object, with data points every 5 minutes
    # We can choose between 1 min, 5 min, and 1 hour intervals
    # Subtracting the microsecond is so the data will include the midnight data variable. Otherwise we are missing the first piece of data
    full_omni_data = omni.hro2_5min(ti - dt.timedelta(microseconds=1), te)

    # Convert the time series object to a pandas dataframe
    full_omni_data = full_omni_data.to_dataframe()

    # The first value of every month appears twice (I assume it's a bug). Remove the duplicate value
    full_omni_data = full_omni_data.drop_duplicates()

    # Convert pandas dataframe to an xarray dataset
    full_omni_data = full_omni_data.to_xarray()

    # We only want the V and B values in this xarray. We also want to create new indices where each individual component of V and B will reside
    # Ex: V_index = (Vx, Vy, Vz)
    omni_data = xr.Dataset(
        coords={'Time': full_omni_data['Time'], 'V_index': ['Vx', 'Vy', 'Vz'], 'B_index': ['Bx', 'By', 'Bz']})

    # Concatenate the V and B values so that they will fit inside the above dataset.
    V_values = np.vstack((full_omni_data['Vx'], full_omni_data['Vy'], full_omni_data['Vz'])).T
    B_values = np.vstack((full_omni_data['BX_GSE'], full_omni_data['BY_GSE'], full_omni_data['BZ_GSE'])).T

    # Put the values into the new dataset
    omni_data['V_OMNI'] = xr.DataArray(V_values, dims=['Time', 'V_index'], coords={'Time': omni_data['Time']})
    omni_data['B_OMNI'] = xr.DataArray(B_values, dims=['Time', 'B_index'], coords={'Time': omni_data['Time']})

    # Rename time so that it is the same as the other datasets, making concatenation easier
    omni_data = omni_data.rename({'Time': 'time'})

    return omni_data


def get_dis_data(sc, mode, level, ti, te, binned=False):
    # If binned is true, the first case when binning will not have enough data to bin into 5 minute intervals.
    # But we also don't want values to overlap from day to day, so we have to take away another 2.5 minutes from the end so that we don't see repeats
    if binned == True:
        ti = ti - dt.timedelta(minutes=2.5)
        te = te - dt.timedelta(minutes=2.5)

        # the bin_5min program requires a multiple of 5 minutes from start to end (so no data is left off)
        # check if the time range given is a multiple of 5 minutes, and if it isn't, add onto the end whatever time is needed to make it a 5 minute interval
        if ((te - ti) % dt.timedelta(minutes=5)) / dt.timedelta(seconds=1) != 0:
            te = te + dt.timedelta(
                minutes=(5 - ((te - ti) / dt.timedelta(minutes=5) - int((te - ti) / dt.timedelta(minutes=5))) * 5))

    # Download dis_data
    full_dis_data = fpi.load_moms(sc, mode, level, 'dis-moms', ti, te)

    # Slice the velocity data
    V_data = full_dis_data['velocity']

    # Make a new dataset for the velocity data to go into, with renamed variables
    dis_data = xr.Dataset(coords={'time': full_dis_data['time'], 'V_index': ['Vx', 'Vy', 'Vz']})

    # Place the velocity data into the new dataset
    dis_data['V_DIS'] = xr.DataArray(V_data, dims=['time', 'V_index'], coords={'time': full_dis_data['time']})

    if binned == True:
        dis_data = dm.bin_5min(dis_data, ['V_DIS'], ['V'], ti, te)

    return dis_data


def get_des_data(sc, mode, level, ti, te, binned=False):
    # If binned is true, the first case when binning will not have enough data to bin into 5 minute intervals.
    # But we also don't want values to overlap from day to day, so we have to take away another 2.5 minutes from the end so that we don't see repeats
    if binned == True:
        ti = ti - dt.timedelta(minutes=2.5)
        te = te - dt.timedelta(minutes=2.5)

        # the bin_5min program requires a multiple of 5 minutes from start to end (so no data is left off)
        # check if the time range given is a multiple of 5 minutes, and if it isn't, add onto the end whatever time is needed to make it a 5 minute interval
        if ((te - ti) % dt.timedelta(minutes=5)) / dt.timedelta(seconds=1) != 0:
            te = te + dt.timedelta(
                minutes=(5 - ((te - ti) / dt.timedelta(minutes=5) - int((te - ti) / dt.timedelta(minutes=5))) * 5))

    # Download des_data
    full_des_data = fpi.load_moms(sc, mode, level, 'des-moms', ti, te)

    # Slice the velocity data
    V_data = full_des_data['velocity']

    # Make a new dataset for the velocity data to go into, with renamed variables
    des_data = xr.Dataset(coords={'time': full_des_data['time'], 'V_index': ['Vx', 'Vy', 'Vz']})

    # Place the velocity data into the new dataset
    des_data['V_DES'] = xr.DataArray(V_data, dims=['time', 'V_index'], coords={'time': full_des_data['time']})

    if binned == True:
        des_data = dm.bin_5min(des_data, ['V_DES'], ['V'], ti, te)

    return des_data


# If you are having problems with this function, delete all Kp files in data/kp and run again. This may fix it
def get_kp_data_old(ti, te, expand=[None]):
    # Location of the files on the server
    remote_location = 'ftp://ftp.gfz-potsdam.de/pub/home/obs/Kp_ap_Ap_SN_F107/'
    # Location where the file will be places locally
    local_location = 'data/kp/'

    # Parts of the filename, will be put together along with a year number. final product eg: Kp_ap_2018.txt
    file_name_template = "Kp_ap_"
    file_name_extension = '.txt'

    # Where the list of data points required will be stored
    fname_list = []
    increment = ti.year

    # Making the names of all the required files
    while increment <= te.year:
        fname_list.append(file_name_template + str(increment) + file_name_extension)
        increment += 1

    # If the required files are not already downloaded, download them
    # Choose each individual file name from the list
    download_ftp_files(remote_location, local_location, fname_list)

    # Combine all of the needed files into one dataframe
    full_kp_data = read_txt_files(fname_list, local_location)

    # Select the data we actually want
    time, kp = dm.slice_data_by_time(full_kp_data, ti, te)

    # When you are given certain times and want the associated Kp value, give the times in the expand variable,
    # And expand_kp will give back a list of Kp values at the given times
    if expand[0] != None:
        kp = dm.expand_kp(time, kp, expand)
        time = expand

    kp = kp.astype('float64')

    # I have the option to put in UT here. Not going to rn but could at a later point
    # Create an empty dataset at the time values that we made above
    kp_data = xr.Dataset(coords={'time': time})

    # Put the kp data into the dataset
    kp_data['Kp'] = xr.DataArray(kp, dims=['time'], coords={'time': time})

    return kp_data


def get_IEF_data_old(ti, te, expand=[None]):
    # Note that this only works correctly for one day of info. Maybe will generalize later but for now exactly 1 day
    # Also note that the fgm data should be downloaded with the binned variable set to True

    omni_data = get_omni_data(ti, te)

    # Find the magnitude of the velocity vector of the plasma at each point
    V = np.array([])
    for counter in range(len(omni_data['V_OMNI'].values)):
        start = np.linalg.norm(omni_data['V_OMNI'][counter].values)
        V = np.append(V, [start])

    # Can't really verify that these are right. Just gotta hope I guess (especially w theta). I think it worked correctly
    By = omni_data['B_OMNI'][:, 1].values
    Bz = omni_data['B_OMNI'][:, 2].values
    theta = np.arctan(By / Bz)

    # IEF = V*sqrt(B_y^2 + B_z^2)sin^2(\theta / 2) -> V is the velocity of the plasma, should be in OMNI
    # theta = tan^−1(B_Y/B_Z)
    # The /1000 is from dimensional analysis, to keep units in mV/m
    IEF = V * np.sqrt((By ** 2) + (Bz ** 2)) * (np.sin(theta / 2) ** 2) / 1000

    if expand[0] != None:
        IEF = dm.expand_kp(omni_data['time'].values, IEF.astype('str'), expand)
        time = expand
    else:
        time = omni_data['time'].values

    IEF = IEF.astype('float64')

    # Create an empty dataset at the time values that we made above
    IEF_data = xr.Dataset(coords={'time': time})

    # Put the data into the dataset
    IEF_data['IEF'] = xr.DataArray(IEF, dims=['time'], coords={'time': time})

    return IEF_data


# If you are having problems with this function, delete all Kp files in data/dst and run again. This may fix it
def get_dst_data_old(ti, te, expand=None):
    # I use two different types of data: since this website only has real-time data from 2020 onwards, and provisional data from 2015 to 2019, I have to use two separate links

    # Location of the files on the server. the month/year is part of the link, so the before and after is broken apart
    # example link: https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/202011/index.html for November 2020
    real_time_remote_location_start = 'https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/'
    provisional_remote_location_start = 'https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/'
    remote_location_end = '/index.html'

    # Location where the file will be places locally
    local_location = 'data/dst/'

    # Naming the files
    file_name_template = "Dst_"
    file_name_extension = '.html'

    remote_location_list = []
    local_location_list = []
    increment_month = ti.month
    increment_year = ti.year

    # gets all the data in years leading up to the last desired year
    while increment_year < te.year:
        if increment_year < 2020:
            remote_location_start = provisional_remote_location_start
        else:
            remote_location_start = real_time_remote_location_start
        while increment_month <= 12:
            if increment_month < 10:
                remote_location_list.append(
                    remote_location_start + str(increment_year) + '0' + str(increment_month) + remote_location_end)
                local_location_list.append(local_location + file_name_template + str(increment_year) + '_' + '0' + str(
                    increment_month) + file_name_extension)
            else:
                remote_location_list.append(
                    remote_location_start + str(increment_year) + str(increment_month) + remote_location_end)
                local_location_list.append(local_location + file_name_template + str(increment_year) + '_' + str(
                    increment_month) + file_name_extension)
            increment_month += 1
        increment_month = 1
        increment_year += 1

    # gets all the data in the year equal to the end of the desired times
    while increment_month <= te.month:
        if te.year < 2020:
            remote_location_start = provisional_remote_location_start
        else:
            remote_location_start = real_time_remote_location_start
        if increment_month < 10:
            remote_location_list.append(
                remote_location_start + str(increment_year) + '0' + str(increment_month) + remote_location_end)
            local_location_list.append(local_location + file_name_template + str(increment_year) + '_' + '0' + str(
                increment_month) + file_name_extension)
        else:
            remote_location_list.append(
                remote_location_start + str(increment_year) + str(increment_month) + remote_location_end)
            local_location_list.append(local_location + file_name_template + str(increment_year) + '_' + str(
                increment_month) + file_name_extension)
        increment_month += 1

    # download the data
    download_html_data(remote_location_list, local_location_list)

    # read the data into python
    full_data = read_txt_files(local_location_list, mode='Dst')

    time, dst = dm.slice_dst_data(full_data, ti, te)

    # so apparently the website has values from 1 to 24. 1 is the average of the dst values from midnight to 1am. i had coded this to have this same value at the midnight up to this point
    # shift everything over 30 minutes or 1 hour. 30 minutes works best since it then works as it should with expand
    # since expand_kp assumes the time is given at the center of the bin, doing 30 minutes is probs best. Though I could also include an if statement and do 1 hour if expand is None
    time = np.array(time) + dt.timedelta(minutes=30)

    if expand is not None:
        dst = dm.expand_kp(time, dst, expand)
        time = expand

    dst = dst.astype('float64')  # dst is an integer value, but to keep consistent with kp ill use a float

    # I have the option to put in UT here. Not going to rn but could at a later point
    # Create an empty dataset at the time values that we made above
    dst_data = xr.Dataset(coords={'time': time})

    # Put the kp data into the dataset
    dst_data['DST'] = xr.DataArray(dst, dims=['time'], coords={'time': time})

    return dst_data


def get_symh_data_old(ti, te, expand=None, binned=False):
    # So I couldn't find any way to actually download the data from here, since the data is hidden behind a submit button
    # So in order for this to work you will have to download the data files manually
    # The website for this is https://wdc.kugi.kyoto-u.ac.jp/aeasy/index.html
    # It only has to be done once, but the most recent file will have to be redownloaded if you want to use the most recent data
    # The files go in data/symh. Download all the data for each year, and name them symh_YYYY.

    if expand is not None and binned == True:
        raise ValueError("Expand and binned cannot both be used")

    if binned == True:
        ti = ti - dt.timedelta(minutes=2.5)
        te = te - dt.timedelta(minutes=2.5)

    fname_list = []
    increment = ti.year

    # Making the names of all the required files
    while increment <= te.year:
        fname_list.append('symh_' + str(increment)+'.dat')
        increment += 1

    full_symh_data = read_txt_files(fname_list, local_location='data/symh/', mode='symh')

    time, symh = dm.slice_symh_data(full_symh_data, ti, te, binned=binned)

    if binned==False:
        time = np.array(time) + dt.timedelta(seconds=30)

    if expand is not None:
        symh = dm.expand_kp(time, symh, expand)
        time = expand

    symh = symh.astype('float64')  # symh is an integer value, but to keep consistent with kp ill use a float

    # I have the option to put in UT here. Not going to rn but could at a later point
    # Create an empty dataset at the time values that we made above
    symh_data = xr.Dataset(coords={'time': time})

    # Put the kp data into the dataset
    symh_data['SYMH'] = xr.DataArray(symh, dims=['time'], coords={'time': time})

    return symh_data


def get_aspoc_data(sc, mode, level, start_date, end_date, binned=False):
    aspoc_data = util.load_data(sc=sc, instr='aspoc', mode=mode, level=level,
                                start_date=start_date, end_date=end_date)

    data = (aspoc_data[[sc + '_aspoc_status', sc + '_aspoc_var', sc + '_aspoc_ionc']]
            .rename({'Epoch': 'time',
                     sc + '_aspoc_var': 'dt_minus',
                     sc + '_aspoc_status': 'aspoc_status',
                     sc + '_aspoc_lbl': 'aspoc_lbl',
                     sc + '_aspoc_ionc': 'ion_current'})
            )

    # Set the sample interval as datetimes
    # Note that the times are at the center of the bins
    data['dt_minus'] = data['dt_minus'].astype('timedelta64[ns]')

    if binned==True:
        # doesn't actually bin, since on-off stuff doesn't really make sense when binned. So just remove the data points that aren't at a 5 minute interval
        indices = []
        for counter in range(len(data['time'])):
            string_of_datetime = str(data['time'].values[counter])
            if int(string_of_datetime[17:19])%5==0 and counter !=len(data['time'])-1:
                indices.append(counter)

    data = data.isel(time=indices)

    return data