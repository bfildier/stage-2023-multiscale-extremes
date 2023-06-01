## Import packages and paths
import argparse
import sys
import os
sys.path.insert(0, os.getcwd()+'/src/')
sys.path.insert(0, '/home/mcarenso/code/stage-2023-multiscale-extremes/scripts/src/')
from myImports import *

sim = "SAM"
stringSST = "300"

if sim == "SAM" :     
    file_seg='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2023_05/Dspread0K/irtb/TOOCAN_2.07_SAM_large300_2D_irtb.nc'
    file_tracking='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2023_05/Dspread0K/irtb/FileTracking/TOOCAN-SAM_large300_2D_irtb.dat.gz'
    sim_path = "/bdd/MT_WORKSPACE/MCS/RCE/SAM/INPUTS/v2023_05/SAM_RCE_large300_2D_pr.nc"
    output_path = "/homedata/mcarenso/Stage2023/SAM/"+stringSST+"K/"
    
elif sim == "MESONH":
    file_seg='/bdd/MT_WORKSPACE/MCS/RCE/MESONH/TOOCAN/TOOCAN_v2023_05/Dspread0K/irtb/TOOCAN_2.07_MESONH_large300_2D_irtb.nc'
    file_tracking='/bdd/MT_WORKSPACE/MCS/RCE/MESONH/TOOCAN/TOOCAN_v2023_05/Dspread0K/irtb/FileTracking/TOOCAN-MESONH_large300_2D_irtb.dat.gz'
    sim_path = '/bdd/MT_WORKSPACE/MCS/RCE/MESONH/INPUTS/v2023_05/MESONH_RCE_large300_2D_pr.nc'
    output_path = "/homedata/mcarenso/Stage2023/MESONH/"+stringSST+"K/"

elif sim == "ICON":
    file_seg = '/bdd/MT_WORKSPACE/MCS/RCE/ICON/TOOCAN/TOOCAN_v2023_05/Dspread0K/irtb/TOOCAN_2.07_ICON_large300_2D_irtb.nc'
    file_tracking = '/bdd/MT_WORKSPACE/MCS/RCE/ICON/TOOCAN/TOOCAN_v2023_05/Dspread0K/irtb/FileTracking/TOOCAN-ICON_large300_2D_irtb.dat.gz' 
    sim_path = '/bdd/MT_WORKSPACE/MCS/RCE/ICON/INPUTS/v2023_05/ICON_RCE_large300_2D_pr.nc'
    output_path = "/homedata/mcarenso/Stage2023/ICON/"+stringSST+"K/"
    
    
## Load data
print("Loading data...")
Precip = xr.open_dataset(sim_path)

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

filename = "Max_Precip_per_MCS_data_and_dist"

if os.path.isfile(os.path.join(output_path, filename)):
    # File exists, load the object
    with open(os.path.join(output_path, filename), 'rb') as file:
        dist_Max_Prec = pickle.load(file)
else:
    # File doesn't exist, create the object
    Max_Precip = []  # Initialize the list to store maximum values
    Max_Precip_index = []  # Initialize the list to store first indices

    Max_Precip = []  # Initialize the list to store maximum values
    Max_Precip_index = []  # Initialize the list to store first indices
    Max_Precip_label = [] # Store corresponding MCS label
    
    for label in MCS_2h_to_10h_labels:
        timeinit = MCS_2h_to_10h[MCS_2h_to_10h_labels.index(label)].timeinit
        duration = MCS_2h_to_10h[MCS_2h_to_10h_labels.index(label)].duration
        masked_values = Precip.where(mask_2h_to_10h == label).to_array()  # Apply mask for the current label
        if not np.all(np.isnan(masked_values)):  # If the array is not full of NaN values
            max_value = np.nanmax(masked_values)  # Get the maximum value, handling NaN values
            if not np.isnan(max_value): # Maybe unnecessary there
                max_value_indices = np.unravel_index(np.nanargmax(masked_values), masked_values.shape)  # Get the first index of the maximum value
                Max_Precip_index.append(max_value_indices)
                Max_Precip_label.append(label)
                Max_Precip.append(max_value)
                
    dist_Max_Prec = cs.Distribution(name="SAM Precipitation", bintype = "invlogQ", nd = 4, fill_last_decade=True)
    dist_Max_Prec.computeDistribution(sample = Max_Precip)
    dist_Max_Prec.storeSamplePoints(sample =  Max_Precip, sizemax = int(1e7))

    dist_and_data = (dist_Max_Prec, Max_Precip, Max_Precip_index, Max_Precip_label)
    
    # Save the object as a file
    with open(os.path.join(output_path, filename), 'wb') as file:
        pickle.dump(dist_and_data, file)

from myFuncs import Age
Ages_of_MaxPrecip_over_bins = [[] for _ in dist_Max_Prec.bin_locations]

for i, bin_loc in enumerate(dist_Max_Prec.bin_locations):
    for idx in bin_loc:
        label = Max_Precip_label[idx]
        time = Max_Precip_index[idx][0]
        
        Ages_of_MaxPrecip_over_bins[i].extend(Age(label, time, bin_loc, MCS_2h_to_10h[idx_by_label(label)]))
    
    