import argparse
import numpy as np
import xarray as xr
import os 

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('temp', type=str, help='temperature of the simulation. choice between 295, 300 and 305')
parser.add_argument('diag', type=str, help='Name of the diagnostic to deploy in te fine-grid model output')
parser.add_argument('days', type=int, help='Number of days to use for slicing the data')
parser.add_argument('bool_MCS_surf', type=bool, nargs ='?', help='bool used to know fi the MCS label must ba applied to data before using diag')
parser.add_argument('path', type=str, nargs ='?', help='path to the desired .nc file if you want to override SAM["Prec"]')
parser.add_argument('var', type =str, nargs ='?', help='Variable to be extracted from the file specified by path')
#parser.add_argument('dim', type=str, help='either 2D or 3D simulation. For now only 2D supported')


args = parser.parse_args()

# Load the netCDF file
if args.path : 
    ds =xr.open_dataset(args.path).squeeze("z")
    var = args.var
else:
    sam_dir_path = "/bdd/MT_WORKSPACE/REMY/RCEMIP/SAM/"
    path = sam_dir_path+args.temp+"K/rcemip_large_2048x128x74_3km_12s_"+args.temp+"K_64.2Dcom_1.nc"

    # Check if the second file is required
    if args.days > 20 : print("TODO : update code to also open the second .nc")
    ds = xr.open_dataset(path)
    var = "Prec"

# Slice the data
ds = ds.isel(time=slice(48*args.days))
ds["x"] = ((ds["x"])/3e3).astype(int)
ds["y"] = (ds["y"]/3e3).astype(int) 
ds["time"] = np.round(((ds["time"]-75)*48)).astype(int)    


# Coarsen the data
window = {'time': 48, 'x': 32, 'y': 32}
window_mcs = {'time' : 48, 'latitude' : 32, 'longitude' : 32}



if args.diag == "mean":
    if args.bool_MCS_surf == True : 
        from myFuncs import maskMean
        temp = "300"
        file_seg='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2022_04/irtb/TOOCAN_2.07_SAM_RCE_large'+temp+'_2D_irtb.nc'   ## TODO PPass arguemnt temp correctly
        n_days = 20
        label_mask = xr.open_dataarray(file_seg).isel(time=slice(48*n_days)) ## TODO Pass arguemnt n_days correctly
        mask = ~np.isnan(label_mask.values)
        
        da = ds[var].where(mask)
        output = da.coarsen(window).reduce(maskMean)
    else : output = ds[var].coarsen(window).mean()
    
    
    
    
    
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

    MCS_labels = xr.open_dataarray(file_seg).isel(time=slice(48*args.days))
    MCS_labels = MCS_labels.assign_coords({'time' : MCS_labels.time, 'latitude' : MCS_labels.latitude, 'longitude' : MCS_labels.longitude})
    output = MCS_labels.coarsen(window_mcs).reduce(growthRate)


# Save the data to a netCDF file
output_dir = "/home/mcarenso/code/stage-2023-multiscale-extremes/outputs/"
if args.bool_MCS_surf is True : diag_name = args.diag + '_by_MCS_'
output_name = f"rcemip_{var.lower()}_1°x1day_remeshed_by_{diag_name}_with_{args.temp}K_for_{args.days}days.nc"
output.to_netcdf(output_dir+output_name)
if os.path.exists(output_dir+output_name):
    print(f"File {output_name} saved in outputs")
else : print("!!! File not saved as intended, stay alert !!!")
