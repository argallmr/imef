import datetime as dt
from dateutil.relativedelta import relativedelta
from pathlib import Path

import xarray as xr
import numpy as np

# FTP
from ftplib import FTP, error_perm as ftp_error_perm
import pandas as pd

# HTML
from bs4 import BeautifulSoup

# DSCOVR_Downloader
from pathlib import Path
import re
import requests
from tqdm import tqdm
import gzip
import shutil

# import sunpy.time
# from swfo import config
# from swfo.dataprocessing import util

# Sources of DSCOVR data
# https://www.ngdc.noaa.gov/dscovr/next/
# https://www.ngdc.noaa.gov/dscovr/data/

# The file names of interest to you are probably:
# mg0 - mag Level 0 (counts)
# mg1 - mag Level 1 (calibrated physical values)
# m1s - mag Level 2 (1 second average)
# m1m - mag Level 2 (1 minute average)
# vc0 - raw CCSDS frames containing both housekeeping and science

# Remote data locations
ncei_url = 'https://www.ngdc.noaa.gov/dscovr/data/'
kp_ftp_site = 'ftp.gfz-potsdam.de'
kp_ftp_dir = 'pub/home/obs/Kp_ap_Ap_SN_F107/'
dst_realtime_url = 'https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/'
dst_provisional_url = 'https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/'
dst_final_url = 'https://wdc.kugi.kyoto-u.ac.jp/dst_final/'

# Local data locations
data_root = Path('~/data/').expanduser()
dropbox_root = data_root / 'dropbox'
mirror_root = None


class Downloader():
    '''
    A class for downloading a single dataset.

    The following methods must be implemented by sub-classes:
    '''

    def load(self, start_time, end_time):
        raise NotImplementedError

    def local_file(self, interval):
        '''
        Find local file matching the given time interval, regardless of
        version identifier.

        Parameters
        ----------
        interval : tuple of datetime.datetime
            Start and end time associated with a single file

        Returns
        -------
        path : list
            Absolute file paths
        '''

        # Determine if local files have the same time interval
        local_files = self.search_local(interval)
        for file in local_files:
            parts = self.parse_filename(file, to_datetime=True, to_dict=True)
            if ((parts['start_time'] == interval[0])
                    and (parts['stop_time'] == interval[1])):
                return file

        # No file was found
        return None

    def local_path(self, interval, exact=False):
        '''
        Absolute path to a single file.

        Parameters
        ----------
        interval : tuple of datetime.datetime
            Start and end time associated with a single file

        Returns
        -------
        path : str
            Absolute file path
        '''
        local_path = self.local_dir(interval) / self.fname(interval)
        return data_root / local_path

    def local_path_exists(self, interval):
        '''
        Check if a local path exists.

        Parameters
        ----------
        interval : tuple of datetime.datetime
            Start and end time associated with a single file

        Returns
        -------
        exists : bool
            True if local file exists. False otherwise.
        '''
        return self.local_path(interval).exists()

    def intervals(self, start_time, end_time):
        '''
        Break the time interval down into a set of intervals associated
        with individual file names.

        Parameters
        ----------
        start_time, end_time : datetime.datetime
            Start and end times of the data interval

        Returns
        -------
        intervals : list of tuples
            Time intervals (start_time, end_time) associated with individual
            data files
        '''
        pass

    def filter_files_by_time(self, files, interval):
        '''
        Filter file names by a time interval.

        Parameters
        ----------
        files : list of str
            Names of files to be filtered
        interval : tuple of `datetime.datetime`
            Start and end of the time interval in which to keep file names

        Returns
        -------
        result : list
            File names with data in time interval
        '''

        result = []
        for file in files:
            parts = self.parse_filename(file, to_datetime=True, to_dict=True)
            if ((parts['start_time'] <= interval[0])
                    and (parts['stop_time'] >= interval[1])):
                result.append(file)

        return result

    @staticmethod
    def filter_files_by_ptime(files):
        '''
        Filter file names by process time. If there are duplicate file names
        with different process times, the file name with the most recent
        process time will be kept.

        Parameters
        ----------
        files : list of str
            Names of files to be filtered

        Returns
        -------
        result : list
            Filtered results
        '''

        f_out = []
        bases = []
        ptimes = []
        for file in files:
            parts = parse_filename(file, to_dict=True)
            base = '_'.join([value
                             for key, value in parts.items()
                             if key != 'proc_time'])
            ptime = dt.datetime.strptime(parts['proc_time'], '%Y%m%d%H%M%S')

            try:
                idx = bases.index(base)
            except ValueError:
                bases.append(base)
                ptimes.append(ptime)
                f_out.append(file)
                continue

            if ptimes[idx] < ptime:
                bases[idx] = base
                ptimes[idx] = ptime
                f_out[idx] = file

        return f_out

    def fname(self, interval):
        '''
        Create the file name associated with a given interval.

        Note that if a file name contains a version number, process time, etc.
        to indicate file updates/changes, a default value should be used in
        the event one is not provided (by means of an instance property). For
        version numbers, this should be the lowest version number permitted by
        the convention used (e.g. 0.0.0); for process times, the current time.

        Parameters
        ----------
        interval : tuple of datetime.datetime
            Start and end time of the data interval

        Returns
        -------
        filename : str
            File name
        '''
        raise NotImplementedError

    def local_dir(self, interval):
        '''
        Local directory for a given interval. This is relative to the
        PyMMS data directory.

        Parameters
        ----------
        interval : tuple of datetime.datetime
            Start and end time of the data interval

        Returns
        -------
        dir : pathlib.Path
            Local directory
        '''
        pass

    def download(self, interval):
        raise NotImplementedError

    def load_local_file(self, interval):
        raise NotImplementedError

    @staticmethod
    def intervals_daily(start_time, end_time):
        '''
        Break down a time interval into sub-intervals that are one day long.
        Hour, minute, and second fields are ignored if present.

        Parameters
        ----------
        interval : tuple of `datetime.datetime`
            Start and end dates of the time interval

        Returns
        -------
        intervallist : list of `sunpy.time.TimeRange`
            Sub-intervals of duration 23 hours 59 minutes and 59 seconds
        '''
        ndays = (start_time - end_time).days
        if interval[1].time() != dt.time(0):
            ndays += 1
        dates = [start_time.date() + dt.timedelta(days=n)
                 for n in range(0, ndays)]

        intervals = [(dt.datetime.combine(d, dt.time(0)),
                      dt.datetime.combine(d, dt.time(23, 29, 59)))
                     for d in dates]

        return intervals

    @staticmethod
    def intervals_monthly(start_time, end_time):
        '''
        Break down a time interval into sub-intervals that are one month long.
        Day, hour, minute, and second fields are ignored if present.

        Parameters
        ----------
        interval : tuple of `datetime.datetime`
            Start and end dates of the time interval

        Returns
        -------
        intervals : list of `datetime.datetime` tuples
            Sub-intervals of duration one month
        '''
        intervals = []
        time = dt.datetime(start_time.year, start_time.month, 1)
        #  - Add the right end point to the last interval
        end_time += relativedelta(months=+1)

        while time.year != end_time.year or time.month != end_time.month:
            times = (time, time +relativedelta(months=+1) - dt.timedelta(microseconds=1))
            intervals.append(times)
            time += relativedelta(months=+1)

        return intervals

    @staticmethod
    def intervals_yearly(start_time, end_time):
        '''
        Break down a time interval into sub-intervals that are one year long.
        Month, day, hour, minute, and second fields are ignored if present.

        Parameters
        ----------
        interval : tuple of `datetime.datetime`
            Start and end dates of the time interval

        Returns
        -------
        intervals : list of `datetime.datetime` tuples
            Sub-intervals of duration one year
        '''
        # Each year from beginning to end
        #  - Add the right end point to the last interval
        nyears = end_time.year - start_time.year + 2
        dates = [dt.datetime(start_time.year + n, 1, 1, 0)
                 for n in range(0, nyears)]

        # Intervals span from one year up to the next year minus one microsecond
        intervals = [(d0, d1 - dt.timedelta(microseconds=1))
                     for d0, d1 in zip(dates[:-1], dates[1:])]

        return intervals

    @staticmethod
    def parse_filename(filename, to_datetime=False, to_dict=False):
        '''
        break a file name down into its components.

        Parameters
        ----------
        filename : str
            Name of the file to be parsed
        to_datetime : bool
            Convert times in the file name to datetime objects
        to_dict : bool
            Convert the results to a dictionary. This is required by the
            `filter_files_by_time` and `filter_files_by_ptime` methods,
            which count on there being `start_time`, `stop_time`, and
            `proc_time` keys.

        Returns
        -------
        intervallist : list of `sunpy.time.TimeRange`
            Sub-intervals of duration 23 hours 59 minutes and 59 seconds
        '''
        parts = str(Path(filename).stem).split('_')
        if to_datetime:
            raise NotImplementedError
        if to_dict:
            raise NotImplementedError
        return tuple(parts)

    def search(self, mirror=False, dropbox=False):
        '''
        Search for files both locally and remotely.
        Parameters
        ----------
        mirror : bool
            If true, search for files is a read-only mirror of a remote source
        dropbox : bool
            If true, search in a flat dropbox directory used as temporary
            storage for newly created files.
        Returns
        -------
        files : tuple
            Local and remote files within the interval, returned as
            (local, remote), where `local` and `remote` are lists.
        '''

        all_files = []
        roots = [data_root]
        if mirror:
            roots.append(mirror_root)
        if dropbox:
            roots.append(dropbox)

        interval = (self.start_time, self.stop_time)

        # Loop through each directory
        for root in roots:
            # Search for files
            files = self.search_local(interval, root=root,
                                      subdirs=(root != dropbox_root))

            # Combine together
            all_files += files

        # Filter by ptime to remove additional duplicates
        files = filter_files_by_ptime(files)

        return files

    def search_local(self, interval, root=None, subdirs=True, ext='nc',
                     newest=False):
        '''
        Search for files on the local system.
        Parameters
        ----------
        interval : tuple of `datetime.datetime`
            The start and end times of the data interval in which to look
            for data files.
        root : str or path-like
            Root directory in which to search for files. Default is
            swfo.config['data_root']
        subdirs : bool
            If true, search in the subdirectories of `root` as returned by
            `self.local_dir()`.
        ex : str
            The file extension of the files that are the target of the search
        newest : bool
            Select only the most recently processed files
        Returns
        -------
        local_files : list
            Names of the local files
        '''

        # Search the mirror or local directory
        if root is None:
            root = data_root

        # Create all dates between start_date and end_date
        ndays = (interval[1].date() - interval[0].date()).days
        if interval[1].time() != dt.time(0):
            ndays += 1

        if subdirs:
            paths = [root
                     / self.local_dir((interval[0] + dt.timedelta(days=x),
                                       interval[0] + dt.timedelta(days=x)))
                     for x in range(ndays)]
        else:
            paths = [root]

        # Search
        result = []
        for path in paths:
            files = path.glob('*.' + ext)
            result += [path / file for file in files]

        # Remove files that are outside the time interval
        result = self.filter_files_by_time(result, interval)

        # Select only the latest files
        if newest:
            result = self.filter_files_by_ptime(result)

        return result

    def search_remote(self, interval):
        '''
        Search for files on the remote file system.
        Parameters
        ----------
        interval : tuple of `datetime.datetime`
            The start and end times of the data interval in which to look
            for data files.
        Returns
        -------
        remote_files : list
            Names of the remote files
        '''
        raise NotImplementedError


class DSCOVR_Downloader(Downloader):
    mission = 'dscovr'

    def __init__(self, product, start_time, stop_time, pvalue=None):
        self.product = product
        self.start_time = start_time
        self.stop_time = stop_time
        self.pvalue = pvalue

    def download(self, interval):
        '''
        Download a DSCOVR gunzipped netCDF data file. Files are unzipped and the
        original .gz file is deleted.

        Parameters
        ----------
        filename : str
            Name of the file to download
        local_dir : `pathlib.Path`
            Absolute path of directory in which to download the file

        Returns
        -------
        local_file : `pathlib.Path`
            File path
        '''
        local_dir = data_root / self.local_dir(interval)
        filename = self.search_remote(interval)

        # Remote location
        remote_base_url = '/'.join((ncei_url, *local_dir.parts[-2:]))

        local_file = _download_url(remote_base_url, filename, local_dir)

        # Unzip the file
        with gzip.open(local_file, 'rb') as f_in:
            with open(local_file.with_suffix(''), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Delete the gzip file
        local_file.unlink()

        return local_file.with_suffix('')

    def fname(self, interval):
        '''
        Create a DSCOVR file name.

        If `self.pvalue` (the process time) is None, the current time is used.

        Parameters
        ----------
        interval : `sunpy.time.TimeRange`
            Time interval of the data

        Returns
        -------
        fname : str
            Name of the file
        '''

        # Convert times to strings
        #   - If the process time is None, use the current time
        tstart = interval.start.strftime('%Y%m%d%H%M%S')
        tstop = interval.end.strftime('%Y%m%d%H%M%S')
        try:
            tproc = dt.datetime.strftime(self.pvalue, '%Y%m%d%H%M%S')
        except TypeError:
            tproc = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d%H%M%S')

        # Create the file name
        fname = '_'.join((self.product, self.mission,
                          's' + tstart, 'e' + tstop,
                          'p' + tproc, 'pub')) + '.nc'

        return fname

    def search_remote(self, interval):
        '''
        Find a valid file name by searching the NCEI website.

        Parameters
        ----------
        product : str
            The data product
        interval : `sunpy.time.TimeRange`
            Time interval of the data

        Returns
        -------
        fname : str
            The file name
        '''
        tstart = interval.start.strftime('%Y%m%d%H%M%S')
        tstop = interval.end.strftime('%Y%m%d%H%M%S')

        # File name regex pattern. File names are embedded as HTML links.
        #   - Assumes that the interval matches the file name exactly
        fpattern = ('<a href="(oe_{0}_dscovr_s{1}_e{2}_'
                    'p[0-9]+_pub.nc.gz)">'
                    .format(self.product, tstart, tstop)
                    )

        # Search online
        response = requests.get('/'.join((ncei_url,
                                          interval.start.strftime('%Y/%m/'))))

        # Parse the webpage for the file name. Assumes:
        #   - The webpage will maintain only one valid version (process time)
        #   - The file is the first match in the listing
        fname = [match.group(1)
                 for match in re.finditer(fpattern, response.text)]
        return fname[0]

    def intervals(self):
        '''
        Break a time interval down into daily sub-intervals.

        Parameters
        ----------
        starttime, endtime : `datetime.datetime`
            Start and end times of data interval

        Returns
        -------
        intervallist : list of tuples
            Daily time intervals
        '''
        return self.intervals_daily(self.start_time, self.stop_time)

    def local_dir(self, interval):
        '''
        Local data directory relative to the base data path of pymms.

        Parameters
        ----------
        product : str
            The data product
        interval : `sunpy.time.TimeRange`
            Time interval of the data

        Returns
        -------
        local_dir : `pathlib.Path`
            File path relative to `pymms` data tree
        '''
        return (Path(self.mission) / self.product
                / interval.start.strftime('%Y')
                / interval.start.strftime('%m')
                )

    def local_path(self, interval):
        '''
        Absolute file path.

        Parameters
        ----------
        product : str
            The data product
        interval : `sunpy.time.TimeRange`
            Time interval of the data
        pvalue : `datetime.datetime`
            Time following the "p" in the file name. If not given, search for
            the file online.

        Returns
        -------
        path : `pathlib.Path`
            Absolute path to file
        '''
        local_dir = self.local_dir(interval)
        fname = self.fname(interval)

        # Remove the '.gz' extension because files are unzipped when
        # downloaded.
        if self.pvalue is None:
            fname = fname.replace('.gz', '')

        return data_root / local_dir / fname

    def load_file(self, interval):
        '''
        Download and Load data from a locally saved DSCOVR netCDF file into a Dataset

        Parameters
        ----------
        product : str
            The data product
        interval : `sunpy.time.TimeRange`
            Time interval of the data
        pvalue : `datetime.datetime`
            Time following the "p" in the file name. If not given, search for
            the file online.

        Returns
        -------
        ds : `xarray.Dateset`
            Data loaded from file
        '''
        # Find the file to load and check if it has already been downloaded
        #   - Remove the .gz extension on remote file name before searching
        #     locally
        remote_file = self.search_remote(interval)
        local_path = fname(filename=remote_file[0:-3])
        if not local_path.exists():
            local_path = self.download(interval)

        # Load and return the data
        return xr.load_dataset(local_path)

    @property
    def product(self):
        return self._product

    @product.setter
    def product(self, value):
        '''
        Check if a DSCOVR data product name is valid

        Parameters
        ----------
        value : str
            Abbreviation of a DSCOVR data product
        '''

        products = ('att', 'pop'  # attitude and orbital and orientation
                           'f1m', 'f3s', 'fc0', 'fc1',  # faraday cup
                    'm1m', 'm1s', 'mg0', 'mg1',  # magnetometer
                    'rt0', 'rt1', 'vc0', 'vc1')  # telemetry
        if value not in products:
            raise ValueError('Product {0} not in {1}'
                             .format(value, products))
        self._product = value


class Kp_Downloader(Downloader):

    def download(self, interval):
        '''
        Download a Kp data file.

        Parameters
        ----------
        interval : (2) tuple of `datetime.datetime`
            Start and end times of the data to be downloaded

        Returns
        -------
        local_file : `pathlib.Path`
            File path
        '''

        def _ftp_retrlines_callback(line):
            '''Add a newline to each line. It is stripped for the callback'''
            f.write(line + '\n')

        # Check if the file exists
        file_path = self.local_path(interval)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)

        # Stream the text into a file buffer
        with FTP(kp_ftp_site) as ftp:
            ftp.login()
            ftp.cwd(kp_ftp_dir)

            with open(file_path, 'w') as f:
                ftp.retrlines('RETR ' + file_path.name, _ftp_retrlines_callback)

        return file_path

    def fname(self, interval):
        '''
        Create a file name.

        Parameters
        ----------
        interval : (2) tuple of `datetime.datetime`
            Time interval of the data

        Returns
        -------
        fname : str
            Name of the file
        '''
        return 'Kp_ap_' + interval[0].strftime('%Y') + '.txt'

    def intervals(self, start_time, end_time):
        '''
        Break the time interval down into a set of intervals associated
        with individual file names.

        Parameters
        ----------
        start_time, end_time : datetime.datetime
            Start and end times of the data interval

        Returns
        -------
        intervals : list of tuples
            Time intervals (start_time, end_time) associated with individual
            data files
        '''
        return self.intervals_yearly(start_time, end_time)

    def load_file(self, interval):
        '''
        Load a data file. File is downloaded if not found locally.
        '''

        # Download the file if it is not saved locally
        file_path = self.local_path(interval)
        if not file_path.exists():
            file_path = self.download(interval)

        # Read the data
        data = pd.read_table(file_path,
                             delim_whitespace=True,
                             header=0,
                             skiprows=29,
                             names=['Year', 'Month', 'Day', 'Begin-Hour',
                                    'Mid-Hour', 'Begin-Days', 'Mid-Days', 'Kp',
                                    'ap', 'Definitive'])

        # Lambda functions to parse fractional days and hours
        fhr_to_minute = lambda hr: ((hr % 1) * 60).astype(np.int16)
        fday_to_hour = lambda day: ((day % 1) * 24).astype(np.int16)
        fday_to_minute = lambda day: (((day * 24) % 1) * 60).astype(np.int16)

        # Convert the begin times to datetimes
        begin_minutes = fhr_to_minute(data['Begin-Hour'])
        begin_time = [dt.datetime(yr, mo, day, hr, mn)
                      for yr, mo, day, hr, mn in zip(data['Year'], data['Month'],
                                                     data['Day'],
                                                     data['Begin-Hour'].astype(np.int16),
                                                     begin_minutes)]

        # Convert the center times to datetimes
        mid_minutes = fhr_to_minute(data['Mid-Hour'])
        mid_time = [dt.datetime(yr, mo, day, hr, mn)
                    for yr, mo, day, hr, mn in zip(data['Year'], data['Month'],
                                                   data['Day'],
                                                   data['Mid-Hour'].astype(np.int16),
                                                   mid_minutes)]

        # Turn begin days into datetime
        t_epoch = dt.datetime(1932, 1, 1, 0)
        hr_begin = fday_to_hour(data['Begin-Days'])
        min_begin = fday_to_minute(data['Begin-Days'])
        t_begin = [t_epoch + dt.timedelta(days=days, hours=hrs, minutes=mins)
                   for days, hrs, mins in zip(data['Begin-Days'].astype(np.int32),
                                              hr_begin, min_begin)]

        # Turn mid-days into datetimes
        hr_mid = fday_to_hour(data['Mid-Days'])
        min_mid = fday_to_minute(data['Mid-Days'])
        t_mid = [t_epoch + dt.timedelta(days=days, hours=hrs, minutes=mins)
                 for days, hrs, mins in zip(data['Mid-Days'].astype(np.int32),
                                            hr_mid, min_mid)]

        # Turn them in to Series
        begin_time = pd.Series(begin_time)
        mid_time = pd.Series(mid_time)
        t_begin = pd.Series(t_begin)
        t_mid = pd.Series(t_mid)

        # Make sure the time stamps match
        if (begin_time != t_begin).any():
            raise ValueError('Begin times do not match begin elapsed days.')
        if (mid_time != t_mid).any():
            raise ValueError('Center times do not match center elapsed days.')

        # Add the times to the dataframe
        data['time'] = begin_time
        data['t_center'] = mid_time

        # Set time as the index
        data = data.set_index('time')
        return (xr.Dataset.from_dataframe(data)
                .assign_coords({'dt_plus': np.timedelta64(3, 'h'),
                                'dt_minus': np.timedelta64(0, 'h')})
                )

    def local_dir(self):
        '''
        Local data directory relative to the data root.

        Returns
        -------
        local_dir : `pathlib.Path`
            File path relative to the data root
        '''
        return Path('Kp/')

    def local_path(self, interval):
        '''
        Absolute path to a single file.

        Parameters
        ----------
        interval : (2) tuple of datetime.datetime
            Start and end time associated with a single file

        Returns
        -------
        path : `pathlib.Path`
            Absolute file path
        '''
        return data_root / self.local_dir() / self.fname(interval)

    def search_local(self, interval):
        file_path = self.local_path(interval)

        if file_path.exists():
            return file_path
        else:
            return None

    def search_remote(self, interval):

        fname = self.fname(interval)
        with FTP(kp_ftp_site) as ftp:
            ftp.login()
            ftp.cwd(kp_ftp_dir)

            try:
                results = ftp.nlst(fname)[0]
            except ftp_error_perm:
                results = None

        return results


class Dst_Downloader(Downloader):

    def download(self, interval):
        '''
        Download a DST data file.

        Parameters
        ----------
        interval : (2) tuple of `datetime.datetime`
            Start and end times of the data to be downloaded

        Returns
        -------
        local_file : `pathlib.Path`
            File path
        '''
        # Check if the file exists
        file_path = self.local_path(interval)
        # If it doesn't exist, make the directory
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)

        # There are 3 types of DST data, and the data for each year gets updated from time to time
        # Find the most up-to-date version of that data for the given interval, and download that file
        remote_location_list = [dst_final_url, dst_provisional_url, dst_realtime_url]
        location_index=0
        loop=True
        while loop==True:
            remote_location = remote_location_list[location_index] + interval[0].strftime('%Y%m') + '/index.html'
            r = requests.get(remote_location)
            if r.status_code == 200:
                loop=False
            else:
                location_index += 1

        # Write file to local path
        open(file_path, 'wb').write(r.content)

        return file_path

    def fname(self, interval):
        '''
        Create a file name.

        Parameters
        ----------
        interval : (2) tuple of `datetime.datetime`
            Time interval of the data

        Returns
        -------
        fname : str
            Name of the file
        '''
        return 'Dst_' + interval[0].strftime('%Y%m') + '.html'

    def intervals(self, start_time, end_time):
        '''
        Break the time interval down into a set of intervals associated
        with individual file names.

        Parameters
        ----------
        start_time, end_time : datetime.datetime
            Start and end times of the data interval

        Returns
        -------
        intervals : list of tuples
            Time intervals (start_time, end_time) associated with individual
            data files
        '''
        return self.intervals_monthly(start_time, end_time)

    def load_file(self, interval):

        # Download data if it doesn't exist locally already
        file_path = self.local_path(interval)
        if not file_path.exists():
            file_path = self.download(interval)

        # Get data from file
        with open(file_path, 'r') as file:
            soup = BeautifulSoup(file, 'html.parser')
            text_split = soup.get_text().split()

        # Skip the website's header
        index=1
        while text_split[index-1] != 'DAY':
            index+=1
        data=[]

        # Combine all the Dst data into a list
        loop=True
        while index < len(text_split) and loop==True:
            try:
                data.append(int(text_split[index]))
            except:
                try:
                    # There are two cases where things should come in here. Either
                    # a) Sometimes the numbers are not separated by whitespace (eg. -98-105-102). Fix those
                    # or b) the dst page has a footer. If it does it will break in the numbers_list=... line, and the loop will end
                    numbers = text_split[index]
                    numbers_list = numbers.split('-')
                    if numbers_list[0] == '':
                        numbers_list.pop(0)
                    numbers_list = (np.array(numbers_list).astype('int64') * -1).tolist()
                    for number in numbers_list:
                        data.append(number)
                except:
                    loop=False
            index+=1

        # Remove every 25th item in the list. These are the day counters, not Dst data
        del data[0::25]

        # Create a datetime list that contains all the times that this file has data for
        datetimes = []
        current = interval[0]
        while current < interval[1]:
            datetimes.append(current)
            current += dt.timedelta(hours=1)

        dataset = xr.Dataset(coords={'time':datetimes})
        dataset['Dst'] = xr.DataArray(data, dims=['time'], coords={'time': datetimes})
        dataset = dataset.assign_coords({'dt_plus': np.timedelta64(1, 'h'),
                               'dt_minus': np.timedelta64(0, 'h')})

        return dataset

    def local_dir(self):
        '''
        Local data directory relative to the data root.

        Returns
        -------
        local_dir : `pathlib.Path`
            File path relative to the data root
        '''
        return Path('Dst/')

    def local_path(self, interval):
        '''
        Absolute path to a single file.

        Parameters
        ----------
        interval : (2) tuple of datetime.datetime
            Start and end time associated with a single file

        Returns
        -------
        path : `pathlib.Path`
            Absolute file path
        '''
        return data_root / self.local_dir() / self.fname(interval)

    def search_local(self, interval):
        file_path = self.local_path(interval)

        if file_path.exists():
            return file_path
        else:
            return None

    # def search_remote(self, interval):
    # this isn't even used in Kp_downloader. what is it for




def _download_ftp(ftp_base, fname, local_dir):
    raise NotImplementedError


def _download_url(remote_base_url, fname, local_dir):
    '''
    Download a file

    Parameters
    ----------
    remote_base_url : str
        URL at which the file is located
    fname : str
        Name of the file to be downloaded
    local_dir : `pathlib.Path`
        Absolute path of directory in which to save the file

    Returns
    -------
    local_file : `pathlib.Path`
        Absolute path to downloaded file
    '''

    if not local_dir.exists():
        local_dir.mkdir(parents=True)

    remote_file = '/'.join((remote_base_url, fname))
    local_file = local_dir / fname

    r = requests.get(remote_file, stream=True, allow_redirects=True)
    total_size = int(r.headers.get('content-length'))
    initial_pos = 0

    # Download
    with open(local_file, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True,
                  desc=fname, initial=initial_pos,
                  ascii=True) as pbar:

            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return local_file