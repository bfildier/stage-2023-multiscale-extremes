import numpy as np 
import matplotlib.pyplot as plt 
import os
import sys
import xarray as xr

sys.path.insert(0, os.getcwd()+'/../../conditional-stats/src/')
sys.path.insert(0, os.getcwd()+'/../../conditional-stats/plotting/src/')
import conditionalstats as cs
from plot1DInvLog import *
from plot1D import *