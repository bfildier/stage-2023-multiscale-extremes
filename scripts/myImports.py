import numpy as np 
import matplotlib.pyplot as plt 
import os
import sys
import xarray as xr
import dask.array as da
import dask

sys.path.insert(0, os.getcwd()+'/../../conditional-stats/src/')
sys.path.insert(0, os.getcwd()+'/../../conditional-stats/plotting/src/')
import conditionalstats as cs
from plot1DInvLog import *
from plot1D import *
from load_TOOCAN_DYAMOND_modif_Ben import *

## TODO garbage collector