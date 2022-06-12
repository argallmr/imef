from pymms.data import edi, util, fgm, fpi, edp
import numpy as np
import xarray as xr
from heliopy.data import omni
import datetime as dt
import shutil
import urllib.request as request
from contextlib import closing
import pandas as pd
import os
import data_manipulation as dm
import requests

np.set_printoptions(threshold=np.inf)


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
    if mode=='Kp':
        header=29
        footer=0
    elif mode=='symh':
        header=0
        footer=0

    # Combine all of the needed files into one dataframe
    for fname in fname_list:
        if mode=='Dst' and int(fname[13:17]) == 2021 and int(fname[18:20]) >= 8 or mode=='Dst' and int(fname[13:17]) >=2022:
            header=34
            footer=55
        elif mode=='Dst' and int(fname[13:17]) >= 2020:
            header=34
            footer=40
        elif mode=='Dst' and int(fname[13:17]) < 2020:
            header = 28
            footer = 41

        # Read file into a pandas dataframe, and remove the text at the top
        if local_location is not None:
            oneofthem = pd.read_table(local_location + fname, header=header, skipfooter=footer)
        else:
            oneofthem = pd.read_table(fname, header=header, skipfooter=footer)

        if mode=='symh':
            # it turns out that the first 3/4 of the file contains stuff that isn't sym-H. So get rid of it
            oneofthem = oneofthem.iloc[int((3/4)*len(oneofthem)):].reset_index(drop=True)

            # rename the column so that concat will work right. each file reads the first line into the header, so each file has a diff header unless this is done
            oneofthem = oneofthem.rename(columns={oneofthem.columns[0]: '1hoursymh'})

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
            te=te+dt.timedelta(minutes=(5-((te-ti)/dt.timedelta(minutes=5) - int((te-ti)/dt.timedelta(minutes=5)))*5))

    tm_vname = '_'.join((sc, 'edi', 't', 'delta', 'minus', mode, level))

    # Get EDI data
    edi_data = edi.load_data(sc, mode, level, optdesc='efield', start_date=ti, end_date=te)

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
            te=te+dt.timedelta(minutes=(5-((te-ti)/dt.timedelta(minutes=5) - int((te-ti)/dt.timedelta(minutes=5)))*5))

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
            te=te+dt.timedelta(minutes=(5-((te-ti)/dt.timedelta(minutes=5) - int((te-ti)/dt.timedelta(minutes=5)))*5))

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
            te=te+dt.timedelta(minutes=(5-((te-ti)/dt.timedelta(minutes=5) - int((te-ti)/dt.timedelta(minutes=5)))*5))

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
    edp_data = xr.merge([edp_data_fast, edp_data_slow], compat='override')

    # Reorganize the data to sort by time
    edp_data = edp_data.sortby('time')

    # Cast the float32 values to float64 values to prevent overflow when binning
    edp_data['E_EDP'].values = edp_data['E_EDP'].values.astype('float64')

    if binned == True:
        edp_data = dm.bin_5min(edp_data, ['E_EDP'], ['E'], ti, te)

    return edp_data


def get_omni_data(ti, te):
    # Download the omni data as a timeseries object, with data points every 5 minutes
    # We can choose between 1 min, 5 min, and 1 hour intervals
    # Subtracting the microsecond is so the data will include the midnight data variable. Otherwise we are missing the first piece of data
    full_omni_data = omni.hro2_5min(ti-dt.timedelta(microseconds=1), te)

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
            te=te+dt.timedelta(minutes=(5-((te-ti)/dt.timedelta(minutes=5) - int((te-ti)/dt.timedelta(minutes=5)))*5))

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
            te=te+dt.timedelta(minutes=(5-((te-ti)/dt.timedelta(minutes=5) - int((te-ti)/dt.timedelta(minutes=5)))*5))

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
def get_kp_data(ti, te, expand=[None]):
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


def get_IEF_data(ti, te, expand=[None]):
    # Note that this only works correctly for one day of info. Maybe will generalize later but for now exactly 1 day
    # Also note that the fgm data should be downloaded with the binned variable set to True

    omni_data = get_omni_data(ti, te)

    # Find the magnitude of the velocity vector of the plasma at each point
    V = np.array([])
    for counter in range(len(omni_data['V_OMNI'].values)):
        start = np.linalg.norm(omni_data['V_OMNI'][counter].values)
        V = np.append(V, [start])

    # Can't really verify that these are right. Just gotta hope I guess (especially w theta). I think it worked correctly
    By = omni_data['B_OMNI'][:,1].values
    Bz = omni_data['B_OMNI'][:,2].values
    theta = np.arctan(By/Bz)

    # IEF = V*sqrt(B_y^2 + B_z^2)sin^2(\theta / 2) -> V is the velocity of the plasma, should be in OMNI
    # theta = tan^−1(B_Y/B_Z)
    # The /1000 is from dimensional analysis, to keep units in mV/m
    IEF = V*np.sqrt((By**2)+(Bz**2))*(np.sin(theta/2)**2) / 1000

    if expand[0] != None:
        IEF = dm.expand_kp(omni_data['time'].values, IEF.astype('str'), expand)
        time=expand
    else:
        time=omni_data['time'].values

    IEF = IEF.astype('float64')

    # Create an empty dataset at the time values that we made above
    IEF_data = xr.Dataset(coords={'time': time})

    # Put the data into the dataset
    IEF_data['IEF'] = xr.DataArray(IEF, dims=['time'], coords={'time': time})

    return IEF_data

# If you are having problems with this function, delete all dst files in data/dst and run again. This may fix it
def get_dst_data(ti, te, expand=None):
    # Theres a possibility that I don't have to split up between realtime and provisional. The page doesn't have links for real time before 2020, but they may still exist
    # If it's worth it, maybe change all of them to use realtime instead

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
                remote_location_list.append(remote_location_start + str(increment_year) + '0' + str(increment_month) + remote_location_end)
                local_location_list.append(local_location + file_name_template + str(increment_year) + '_' + '0' + str(increment_month) + file_name_extension)
            else:
                remote_location_list.append(remote_location_start + str(increment_year) + str(increment_month) + remote_location_end)
                local_location_list.append(local_location + file_name_template + str(increment_year) + '_' + str(increment_month) + file_name_extension)
            increment_month+=1
        increment_month = 1
        increment_year+=1

    # gets all the data in the year equal to the end of the desired times
    while increment_month <= te.month:
        if te.year < 2020:
            remote_location_start = provisional_remote_location_start
        else:
            remote_location_start = real_time_remote_location_start
        if increment_month < 10:
            remote_location_list.append(remote_location_start + str(increment_year) + '0' + str(increment_month) + remote_location_end)
            local_location_list.append(local_location + file_name_template + str(increment_year) + '_' + '0' + str(increment_month) + file_name_extension)
        else:
            remote_location_list.append(remote_location_start + str(increment_year) + str(increment_month) + remote_location_end)
            local_location_list.append(local_location + file_name_template + str(increment_year) + '_' + str(increment_month) + file_name_extension)
        increment_month += 1

    # download the data
    download_html_data(remote_location_list,local_location_list)

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

    dst = dst.astype('float64') # dst is an integer value, but to keep consistent with kp ill use a float

    # I have the option to put in UT here. Not going to rn but could at a later point
    # Create an empty dataset at the time values that we made above
    dst_data = xr.Dataset(coords={'time': time})

    # Put the kp data into the dataset
    dst_data['DST'] = xr.DataArray(dst, dims=['time'], coords={'time': time})

    return dst_data


# Note that this works fine, but takes up more memory than the other geomagnetic indices, since it is calculated 60x more than dst.
def get_symh_data(ti, te, expand=None, binned=False):
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