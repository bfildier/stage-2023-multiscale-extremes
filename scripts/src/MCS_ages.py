## Import packages and paths
import argparse
import sys
import os
sys.path.insert(0, os.getcwd()+'/src/')
sys.path.insert(0, '/home/mcarenso/code/stage-2023-multiscale-extremes/scripts/src/')
from myImports import *

## define temperature string and number of days
stringSST = "300" ##295, 300 or 305
n_days = 25

## parser to retrieve the name of the simulation
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('sim', type=str, help='Simulation name, either ICON/I or SAM/S or MESONH/M')

args = parser.parse_args()

if args.sim == "SAM" or args.sim == "S" :     
    file_seg='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2022_04/irtb/TOOCAN_2.07_SAM_RCE_large'+stringSST+'_2D_irtb.nc'
    file_tracking='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2022_04/irtb/FileTracking/TOOCAN-SAM_RCE_large'+stringSST+'_2D_irtb.dat.gz'
    sim_path = "/bdd/MT_WORKSPACE/REMY/RCEMIP/SAM/300K/"
    output_path = "/homedata/mcarenso/Stage2023/MCS_ages/SAM/"+stringSST+"K/"
    
elif args.sim == "MESONH" or args.sim == "M":
    file_seg='/bdd/MT_WORKSPACE/MCS/RCE/MESONH/TOOCAN/TOOCAN_v2023_01/irtb/TOOCAN_2.07_MESONH_RCE_large'+stringSST+'_2D_irtb.nc'
    file_tracking='/bdd/MT_WORKSPACE/MCS/RCE/MESONH/TOOCAN/TOOCAN_v2023_01/irtb/FileTracking/TOOCAN-MESONH_RCE_large'+stringSST+'_2D_irtb.dat.gz'
    sim_path = ''
    output_path = "/homedata/mcarenso/Stage2023/MCS_ages/MESONH/"+stringSST+"K/"

elif args.sim == "ICON" or args.sim == "I":
    file_seg = '/bdd/MT_WORKSPACE/MCS/RCE/ICON/TOOCAN/TOOCAN_v2023_01/irtb/TOOCAN_2.07_ICON_RCE_large'+stringSST+'_2D_irtb.nc'
    file_tracking = '/bdd/MT_WORKSPACE/MCS/RCE/ICON/TOOCAN/TOOCAN_v2023_01/irtb/FileTracking/TOOCAN-ICON_RCE_large'+stringSST+'_2D_irtb.dat.gz' 
    sim_path = ''
    output_path = "/homedata/mcarenso/Stage2023/MCS_ages/MESONH/"+stringSST+"K/"
