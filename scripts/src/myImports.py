## Fundamentals and File Management
import os
import sys
import pickle 

## Math and Data Manipulation
import math 
import numpy as np 
import xarray as xr
import dask.array as da
import bisect
import pandas as pd
from collections import defaultdict

sys.path.insert(0,'/home/mcarenso/code/conditional-stats/src')
sys.path.insert(0,'/home/mcarenso/code/conditional-stats/plotting/src')
sys.path.insert(0,'/home/mcarenso/code/stage-2023-multiscale-extremes/scripts/src')

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

## Ben thermo constants for thermo funcs
from thermoConstants import *

## Illegal stuff
import warnings
import gc




