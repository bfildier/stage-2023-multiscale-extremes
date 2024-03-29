import argparse
import numpy as np
import xarray as xr
import os 

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('temp', type=str, help='temperature of the simulation. choice between 295, 300 and 305')
parser.add_argument('diag', type=str, help='Name of the diagnostic to deploy in te fine-grid model output')
parser.add_argument('bool_MCS_surf', type=bool, nargs ='?', help='bool used to know if the MCS label must ba applied to data before using diag')
parser.add_argument('path', type=str, nargs ='?', help='path to the desired .nc file if you want to override SAM["Prec"]')
parser.add_argument('var', type =str, nargs ='?', help='Variable to be extracted from the file specified by path')
#parser.add_argument('dim', type=str, help='either 2D or 3D simulation. For now only 2D supported')


args = parser.parse_args()

#TOOCAN segmentation masks (les labels des objets MCS, dans la grille originale x,y,t)
file_seg='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2022_04/irtb/TOOCAN_2.07_SAM_RCE_large'+args.temp+'_2D_irtb.nc'

# TOOCAN objects (list d'objets MCS, leur labels et leur caractéristiques)
file_tracking='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2022_04/irtb/FileTracking/TOOCAN-SAM_RCE_large'+args.temp+'_2D_irtb.dat.gz'

sam_dir_path = "/bdd/MT_WORKSPACE/REMY/RCEMIP/SAM/300K/"

# Load the netCDF file
if args.path : 
    ds =xr.open_dataset(args.path).squeeze("z")
    var = args.var
else:
    # Open native precip datasets
    ds1 = xr.open_dataset(sam_dir_path+"rcemip_large_2048x128x74_3km_12s_"+args.temp+"K_64.2Dcom_1.nc")
    ds2 = xr.open_dataset(sam_dir_path+"rcemip_large_2048x128x74_3km_12s_"+args.temp+"K_64.2Dcom_2.nc")

    # Combine datasets
    ds = xr.concat([ds1, ds2], dim='time')
    # Rename dimensions
    ds["x"] = ((ds["x"])/3e3).astype(int)
    ds["y"] = (ds["y"]/3e3).astype(int) 
    ds["time"] = np.round(((ds["time"]-75)*48)).astype(int)
    print(ds)
# Coarsen the data
window = {'time': 48, 'x': 32, 'y': 32}
window_mcs = {'time' : 48, 'latitude' : 32, 'longitude' : 32}



if args.diag == "mean":
    if args.bool_MCS_surf == True : 
        from myFuncs import maskMean
        file_seg='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2022_04/irtb/TOOCAN_2.07_SAM_RCE_large'+args.temp+'_2D_irtb.nc'   ## TODO Pass arguemnt temp correctly
        label_mask = xr.open_dataarray(file_seg)
        mask = ~np.isnan(label_mask.values)
        
        da = ds[var].where(mask)
        output = da.coarsen(window).reduce(maskMean)
    else : 
        print("actually working...")
        output = ds[var].coarsen(window).mean()
       
elif args.diag == "sum" : 
    output = ds[var].coarsen(window).sum()
    
elif args.diag == "max" :
    if args.bool_MCS_surf == True : 
        from myFuncs import maskMax
        output = ds[var].coarsen(window).reduce(maskMax)
    else : output = ds[var].coarsen(window).max()
    
elif args.diag == "truncMean":
    ds_Prec_truncMean = ds[var].where(ds[var] != 0, np.nan)
    output = ds_Prec_truncMean.coarsen(window).mean(skipna=True)

elif args.diag == "fracPrec":
    from myFuncs import condFracPrec
    output = ds[var].coarsen(window).reduce(condFracPrec, keep_attrs=True).rename('FracPrec')

elif args.diag == "growthRate":
    from myFuncs import growthRate

    file_seg='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2022_04/irtb/TOOCAN_2.07_SAM_RCE_large'+args.temp+'_2D_irtb.nc'

    MCS_labels = xr.open_dataarray(file_seg)
    MCS_labels = MCS_labels.assign_coords({'time' : MCS_labels.time, 'latitude' : MCS_labels.latitude, 'longitude' : MCS_labels.longitude})
    output = MCS_labels.coarsen(window_mcs).reduce(growthRate)


# Save the data to a netCDF file
output_dir = "/homedata/mcarenso/Stage2023/SAM/300K/remeshed_variables"
if args.bool_MCS_surf is True : diag_name = args.diag + '_by_MCS_'
output_name = f"rcemip_{var.lower()}_1°x1day_remeshed_by_{diag_name}_with_{args.temp}K.nc"
output.to_netcdf(output_dir+output_name)
if os.path.exists(output_dir+output_name):
    print(f"File {output_name} saved in outputs")
else : print("!!! File not saved as intended, stay alert !!!")
