import argparse
import datetime as dt
import imef.data.database as db

def main():
    # import warnings
    # warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description='Download data required by the IMEF model and save to netCDF.'
    )

    parser.add_argument('sc',
                        type=str,
                        help='Spacecraft Identifier')

    parser.add_argument('start_date',
                        type=str,
                        help='Start date of the data interval: '
                             '"YYYY-MM-DDTHH:MM:SS""'
                        )

    parser.add_argument('end_date',
                        type=str,
                        help='Start date of the data interval: '
                             '"YYYY-MM-DDTHH:MM:SS""'
                        )

    parser.add_argument('-dt', '--sample_interval',
                        default=5,
                        type=float,
                        help='Time interval at which to resample the data',
                        )

    args = parser.parse_args()
    t0 = dt.datetime.strptime(args.start_date, '%Y-%m-%dT%H:%M:%S')
    t1 = dt.datetime.strptime(args.end_date, '%Y-%m-%dT%H:%M:%S')
    dt_out = dt.timedelta(microseconds=int(args.sample_interval * 1e6))

    mode = 'srvy'
    level = 'l2'

    fname = db.multi_interval(args.sc, mode, level, t0, t1, dt_out=dt_out)
    print(fname)


if __name__ == '__main__':
    main()