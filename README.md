# Installation

## Creating a virtual environment

### pyenv

Create environment (only if it doesn't already exist)
```
python -m venv myenv
```

Activate environment
```
.\myenv\Scripts\activate.bat
```

Deactivate environment
```
.\myenv\Scripts\deactivate.bat
```

### Conda
Create an environment
```
conda create -n <environment_name>
```

Activate the environment
```
conda activate <environment_name>
```

Deactivate the environment
```
conda deactivate <environment_name>
```

## Install

### PIP
Install an editable version of the package with PIP. The requirements for this are already accounted for in the `pypackage.toml` file described in PEP 617 and 618. `setuptools` supports these PEPs as of v61.0. It, in turn, requires python>3.8 and pip>=21.

```
cd [...]\imef\
python3 -m pip install -e .
```

In the process, if you see a `WARNING: There was an error checking the latest version of pip.`, follow [these instructions](https://stackoverflow.com/a/77298334)

# Examples

## Gathering Data
These examples show how to gather data.

### Whole Time Interval
This example obtains one day's worth of spacecrat potential data from MMS1 and resamples it to the 5s cadence of EDI. The time interval can be of any duration. To obtain data from all instruments, change `'scpot'` to `'all'`.

```python
import datetime as dt
from imef.data import database as db

# Define the time interval and sample cadence
t0 = dt.datetime(2020, 6, 30, 0, 0)
t1 = dt.datetime(2020, 7, 1, 0, 0)
dt_out = dt.timedelta(seconds=5)

# Get the data
data = db.one_interval('mms1', 'scpot', 'srvy', 'l2', t0, t1, dt_out=dt_out)
```