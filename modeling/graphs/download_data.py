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


def download_ftp_files(remote_location, local_location, fname_list):
    for fname in fname_list:
        # Check if they exist
        if os.path.isfile(local_location + fname) == 0:
            # If they do not exist, create the new file and copy down the data
            # Note that this will not update existing files. May want to figure that out at some point
            with open(local_location + fname, 'wb') as f:
                with closing(request.urlopen(remote_location + fname)) as r:
                    shutil.copyfileobj(r, f)


def read_txt_files(fname_list):
    # Combine all of the needed files into one dataframe
    for fname in fname_list:
        # Read file into a pandas dataframe, and remove the text at the top
        oneofthem = pd.read_table('data/kp/' + fname, header=29)
        if fname == fname_list[0]:
            # If this is the first time going through the loop, designate the created dataframe as where all the data will go
            full_data = oneofthem
        else:
            # Otherwise, combine the new data with the existing data
            full_data = pd.concat([full_data, oneofthem], ignore_index=True)

    return full_data


def get_edi_data(sc, mode, level, ti, te, binned=False):
    # If binned is true, the first case when binning will not have enough data to bin into 5 minute intervals.
    # But we also don't want values to overlap from day to day, so we have to take away another 2.5 minutes from the end so that we don't see repeats
    if binned == True:
        ti = ti - dt.timedelta(minutes=2.5)
        te = te - dt.timedelta(minutes=2.5)

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

    # We want to use E_GSE, so we bin through E_GSE.
    # Note that binning results in the rest of the variables being dropped.
    if binned == True:
        edi_data = dm.bin_5min(edi_data, ['E_GSE'], ['E'], ti, te)

    return edi_data


def get_fgm_data(sc, mode, ti, te, binned=False):
    # If binned is true, the first case when binning will not have enough data to bin into 5 minute intervals.
    # But we also don't want values to overlap from day to day, so we have to take away another 2.5 minutes from the end so that we don't see repeats
    if binned == True:
        ti = ti - dt.timedelta(minutes=2.5)
        te = te - dt.timedelta(minutes=2.5)

    # Get FGM data
    fgm_data = fgm.load_data(sc=sc, mode=mode, start_date=ti, end_date=te)

    if binned == True:
        fgm_data = dm.bin_5min(fgm_data, ['B_GSE'], ['b'], ti, te)

    return fgm_data


def get_mec_data(sc, mode, level, ti, te, binned=False):
    # If binned is true, the first case when binning will not have enough data to bin into 5 minute intervals.
    # But we also don't want values to overlap from day to day, so we have to take away another 2.5 minutes from the end so that we don't see repeats
    if binned == True:
        ti = ti - dt.timedelta(minutes=2.5)
        te = te - dt.timedelta(minutes=2.5)

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
    # If binned is true, the first case when binning will not have enough data to bin into 5 minute intervals.
    # But we also don't want values to overlap from day to day, so we have to take away another 2.5 minutes from the end so that we don't see repeats
    if binned == True:
        ti = ti - dt.timedelta(minutes=2.5)
        te = te - dt.timedelta(minutes=2.5)

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

    # Combine fast and slow data
    edp_data = xr.merge([edp_data_fast, edp_data_slow])

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
    full_kp_data = read_txt_files(fname_list)

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


def get_IEF(fgm_data, ti, te, expand=[None]):
    # Note that this only works correctly for one day of info. Maybe will generalize later but for now exactly 1 day
    # Also note that the fgm data should be downloaded with the binned variable set to True

    # VALUES ARE INCORRECT. FIGURE OUT WHY

    omni_data = get_omni_data(ti, te)

    # Find the magnitude of the velocity vector of the plasma at each point
    V = np.array([])
    for counter in range(len(omni_data['V_OMNI'].values)):
        start = np.linalg.norm(omni_data['V_OMNI'][counter].values)
        V = np.append(V, [start])

    # Can't really verify that these are right. Just gotta hope I guess (especially w theta). I think it worked correctly
    By = fgm_data['B_GSE'][:,1].values
    Bz = fgm_data['B_GSE'][:,2].values
    theta = np.arctan(By/Bz)

    # IEF = V*sqrt(B_y^2 + B_z^2)sin^2(\theta / 2) -> V is the velocity of the plasma, should be in OMNI
    # theta = tan^−1(B_Y/B_Z)
    IEF_data = V*np.sqrt((By**2)+(Bz**2))*(np.sin(theta/2)**2)

    if expand[0] != None:
        IEF_data = dm.expand_kp(omni_data['time'], IEF_data, expand)

    return IEF_data
