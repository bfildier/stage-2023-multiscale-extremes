## Fundamentals and File Management
import os
import sys
import pickle 
import math 

## Math and Data Manipulation
import numpy as np 
import xarray as xr
import dask.array as da

sys.path.insert(0,'/home/mcarenso/code/conditional-stats/src')
sys.path.insert(0,'/home/mcarenso/code/conditional-stats/plotting/src')

import conditionalstats as cs

## Plotting
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, LogNorm

from plot1DInvLog import *
from plot1D import *
from plot2D import *

## dask 
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
import dask.distributed as ddistributed
import dask.delayed as delayed

## ML and stats 
sys.path.insert(0, "/home/mcarenso/.conda/envs/PyLMD/lib/python3.8/site-packages/lmfit/")
from lmfit import Model, Parameters, report_fit
from scipy.interpolate import griddata, LinearNDInterpolator
from bokeh import *

## TODO garbage collector to free up the memory :(




