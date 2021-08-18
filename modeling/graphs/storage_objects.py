import xarray as xr
import datetime as dt
from dataclasses import dataclass


@dataclass
class LandMLT(object):
    L_range: tuple
    MLT_range: tuple
    dL: int
    dMLT: int
    L: xr.DataArray
    MLT: xr.DataArray
    nL: int
    nMLT: int


@dataclass
class DownloadParameters(object):
    sc: str
    mode: str
    level: str
    t0: dt.datetime
    t1: dt.datetime
