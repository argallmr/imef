import datetime as dt
from download_data import download_data_3d
from data_manipulation import average_data_3d

def collect_3d_data():
    # Start and End dates for download. Can be made into arguments if you want
    start_date = dt.datetime(2015, 9, 10)
    end_date = dt.datetime(2017, 12, 15)

    # Whether the output file has been created or not.
    created_file = False

    while start_date < end_date:
        t0 = start_date

        start_date += dt.timedelta(days=1)
        # For some reason when days is increased, many L and MLT values from download_data become NaN.
        # Goes from 8 to 100's/1000's of values
        t1 = start_date

        # If the timedelta increases t1 past the desired end date, make t1 the desired end date
        if t1 > end_date:
            t1 = end_date

        print(t0, "%%", t1)

        # try is in case data fails to load (most likely due to empty files)
        try:
            # Download a pandas dataframe containing electric field data in cartesian coordinates,
            # and azimuthal (MLT), and radial distance of spacecraft
            # Azimuthal is in terms of magnetic local hours
            # Altitude is in an angle in terms of degrees (-90 to 90)
            # Radial distance is roughly equivalent to the radial distance away from Earth in units of Earth radii.
            new_data = download_data_3d("mms1", t0, t1)
        except Exception as ex:
            print("Failed to load", t0, "to", t1)
        else:
            created_file = average_data_3d(new_data, created_file)

        #t0 += dt.timedelta(days=1)
        #start_date = t0

        start_date = t1


if __name__ == '__main__':
    collect_3d_data()