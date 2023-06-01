## Import packages and paths
import argparse
import sys
import os
sys.path.insert(0, os.getcwd()+'/src/')
sys.path.insert(0, '/home/mcarenso/code/stage-2023-multiscale-extremes/scripts/src/')
from myImports import *

## define temperature string and number of days
stringSST = "300" ##295, 300 or 305

## parser to retrieve the name of the simulation
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('sim', type=str, help='Simulation name, either ICON/I or SAM/S or MESONH/M')

args = parser.parse_args()

if args.sim == "SAM" :     
    file_seg='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2023_05/Dspread0K/irtb/TOOCAN_2.07_SAM_large300_2D_irtb.nc'
    file_tracking='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2023_05/Dspread0K/irtb/FileTracking/TOOCAN-SAM_large300_2D_irtb.dat.gz'
    sim_path = "/bdd/MT_WORKSPACE/MCS/RCE/SAM/INPUTS/v2023_05/SAM_RCE_large300_2D_pr.nc"
    output_path = "/homedata/mcarenso/Stage2023/SAM/"+stringSST+"K/"
    
elif args.sim == "MESONH":
    file_seg='/bdd/MT_WORKSPACE/MCS/RCE/MESONH/TOOCAN/TOOCAN_v2023_05/Dspread0K/irtb/TOOCAN_2.07_MESONH_large300_2D_irtb.nc'
    file_tracking='/bdd/MT_WORKSPACE/MCS/RCE/MESONH/TOOCAN/TOOCAN_v2023_05/Dspread0K/irtb/FileTracking/TOOCAN-MESONH_large300_2D_irtb.dat.gz'
    sim_path = '/bdd/MT_WORKSPACE/MCS/RCE/MESONH/INPUTS/v2023_05/MESONH_RCE_large300_2D_pr.nc'
    output_path = "/homedata/mcarenso/Stage2023/MESONH/"+stringSST+"K/"

elif args.sim == "ICON":
    file_seg = '/bdd/MT_WORKSPACE/MCS/RCE/ICON/TOOCAN/TOOCAN_v2023_05/Dspread0K/irtb/TOOCAN_2.07_ICON_large300_2D_irtb.nc'
    file_tracking = '/bdd/MT_WORKSPACE/MCS/RCE/ICON/TOOCAN/TOOCAN_v2023_05/Dspread0K/irtb/FileTracking/TOOCAN-ICON_large300_2D_irtb.dat.gz' 
    sim_path = '/bdd/MT_WORKSPACE/MCS/RCE/ICON/INPUTS/v2023_05/ICON_RCE_large300_2D_pr.nc'
    output_path = "/homedata/mcarenso/Stage2023/ICON/"+stringSST+"K/"

##get sim_path files : 
Precip = xr.open_dataarray(sim_path)

## build&save or load distribution
nd = 5
filename = "Distribution_Precip_"+str(nd)+"decades.pkl"

if os.path.isfile(os.path.join(output_path, filename)):
    # File exists, load the object
    with open(os.path.join(output_path, filename), 'rb') as file:
        dist_Prec = pickle.load(file)
else:
    # File doesn't exist, create the object
    dist_Prec = cs.Distribution(name="SAM Precipitation", bintype = "invlogQ", nd = 5, fill_last_decade=True)
    dist_Prec.computeDistribution(sample = Precip.to_numpy().flatten())
    #dist_SAM_Prec.storeSamplePoints(sample = da.stack(flat=('time', 'y', 'x')).to_numpy(), sizemax = int(1e7))

    # Save the object as a file
    with open(os.path.join(output_path, filename), 'wb') as file:
        pickle.dump(dist_Prec, file)
        
        
## Import MCS list and prepare label list
from load_TOOCAN_DYAMOND_modif_BenAndMax import load_TOOCAN_DYAMOND
MCS = load_TOOCAN_DYAMOND(file_tracking)
MCS_labels = [MCS[i].label for i in range(len(MCS))]

## function to retrieve the indexes in MCS by MCS labels, could be put in myFuncs but need label_list from the tracking file
def idx_by_label(labels, label_list = MCS_labels):
    idxs = [label_list.index(label) for label in labels]
    return idxs

MCS_2h_to_10h = [MCS[i] for i in range(len(MCS)) if MCS[i].duration in np.arange(4, 21, 1).astype(int).tolist()]
MCS_2h_to_10h_labels = [MCS_2h_to_10h[i].label for i in range(len(MCS_2h_to_10h))]
        
## label_mask contains the label of the MCS over the map
label_mask = xr.open_dataarray(file_seg, chunks = {'time' :48, 'longitude' : 32, 'latitude' : 32}).isel(time=slice(48*25)).astype(int) 

## adapt label_mask to 2h_10h MCS
label_2h_to_10h_mask = label_mask.where(label_mask.isin(MCS_2h_to_10h_labels))

mask_2h_to_10h = ~label_2h_to_10h_mask.where(label_2h_to_10h_mask.isnull(), False).isnull()

from myFuncs import Age_vec

output_file = "AgeAnalysis.pkl"
if os.path.isfile(output_path + output_file):
    with open(os.path.join(output_path, output_file), 'rb') as file:
        output = pickle.load(file)
        
else : 
    ## compute the age analysis
    output = dist_Prec.computeAgeAnalysisOverBins(sample = Precip.to_numpy().flatten(), MCS_list = MCS_2h_to_10h, label = label_2h_to_10h_mask.compute().values, sizemax = int(1e6))
    
    ## save the output
    with open(os.path.join(output_path, output_file), 'wb') as file:
        pickle.dump(output, file) 

import pandas as pd
import numpy as np

# Sample data
Ages_over_bins, Ages_of_Xprecip, Xprecip_over_ages, Xprecip_counts, Ages_per_duration, Ages_of_MaxPrecip, MaxPrecip_over_ages, MaxPrecip_MCS_counts = output

x = dist_Prec.ranks
MCS_duration_2h_to_10h = np.arange(2, 10.5, 1/2)

# Create DataFrames for each dataset
df1 = pd.DataFrame({'bin_ranks': x, 'Ages_over_bins': Ages_over_bins})
df2 = pd.DataFrame({'Ages_of_Xprecip': Ages_of_Xprecip, 'Xprecip_over_ages': Xprecip_over_ages})
df3 = pd.DataFrame({'Ages_of_Xprecip': Ages_of_Xprecip, 'Xprecip_counts': Xprecip_counts})
df4 = pd.DataFrame({'MCS_duration_2h_to_10h': MCS_duration_2h_to_10h, 'Ages_per_duration': Ages_per_duration})
df5 = pd.DataFrame({'Ages_of_MaxPrecip': Ages_of_MaxPrecip, 'MaxPrecip_over_ages': MaxPrecip_over_ages})
df6 = pd.DataFrame({'Ages_of_MaxPrecip': Ages_of_MaxPrecip, 'MaxPrecip_MCS_counts': MaxPrecip_MCS_counts})


# Save DataFrames to a CSV file to correct output_path
df1.to_csv(os.path.join(output_path,'ages_over_bins.csv'),header = 'boxplot Ages_over_bins',  index=False)
df2.to_csv(os.path.join(output_path,'xprecip_over_ages.csv'), header = 'Plot as moving average of Xprecip over Xprecip_ages', index=False)
df3.to_csv(os.path.join(output_path,'xprecip_counts.csv'), header = 'Counts of MCS used to compute the Xprecip_over_ages, plot over Ages of Xprecip', index=False)
df4.to_csv(os.path.join(output_path, 'ages_per_duration.csv'), header = 'Boxplot of Ages_per_durations over MCS_duration_2h_10h', index=False)
df5.to_csv(os.path.join(output_path, 'maxprecip_over_ages.csv'), header = 'Plot as moving average of MaxPrecip over MaxPrecip_ages', index=False)
df6.to_csv(os.path.join(output_path, 'maxprecip_MCS_counts.csv'), header = 'Counts of MCS used to compute the MaxPrecip_over_ages, plot over Ages of MaxPrecip', index=False)

print("Data saved to CSV files.")