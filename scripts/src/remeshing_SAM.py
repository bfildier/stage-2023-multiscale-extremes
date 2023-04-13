import argparse
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('temp', type=str, help='temperature of the simulation. choice between 295, 300 and 305')
parser.add_argument('diag', type=str, help='Name of the diagnostic to deploy in te fine-grid model output')
parser.add_argument('days', type=int, help='Number of days to use for slicing the data')
#parser.add_argument('dim', type=str, help='either 2D or 3D simulation. For now only 2D supported')


args = parser.parse_args()

# Load the netCDF file
sam_dir_path = "/bdd/MT_WORKSPACE/REMY/RCEMIP/SAM/"
path = sam_dir_path+args.temp+"K/"+"rcemip_large_2048x128x74_3km_12s_300K_64.2Dcom_1.nc"

# Check if the second file is required
if args.days > 20 : print("TODO : update code to also open the second .nc")
ds = xr.open_dataset(path)

# Slice the data
ds = ds.isel(time=slice(48*args.days)).isel(x=slice(48, -16))
ds["x"] = ((ds["x"]-9.6e4)/3e3).astype(int)
ds["y"] = (ds["y"]/3e3).astype(int) 
ds["time"] = np.round(((ds["time"]-75)*48)).astype(int)         

# Coarsen the data
window = {'time': 48, 'x': 32, 'y': 32}
if args.diag == "mean":
    output = ds["Prec"].coarsen(window).mean() ## *48 same as summing over time axis. 
elif args.diag == "sum" : 
    output = ds["Prec"].coarsen(window).sum()
elif args.diag == "max" :
    output = ds["Prec"].coarsen(window).max()

# Save the data to a netCDF file
output_name = f"rcemip_prec_1°x1day_remeshed_by_{args.diag}_with_{args.temp}K_for_{args.days}days.nc"
output.to_netcdf(f"/home/mcarenso/code/stage-2023-multiscale-extremes/outputs/rcemip_prec_1°x1day_remeshed_by_{args.diag}_with_{args.temp}K_for_{args.days}days.nc")
print(f"File {output_name} saved in outputs")
