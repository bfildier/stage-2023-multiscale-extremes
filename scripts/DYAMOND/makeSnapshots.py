import sys
import os
import argparse
sys.path.insert(0, os.getcwd()+'/src/')
sys.path.insert(0, '/home/mcarenso/code/stage-2023-multiscale-extremes/scripts/src/')
from myImports import *

## define temperature string and number of days
stringSST = "300" ##295, 300 or 305

## parser to retrieve the name of the simulation
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('start', type=int, help='Simulation name, either ICON/I or SAM/S or MESONH/M')
parser.add_argument('end', type=int, help='Simulation name, either ICON/I or SAM/S or MESONH/M')

args = parser.parse_args()


from ProcessingMCS import ProcessMCS
Re = ProcessMCS()

timelag = 0
my_path = '/homedata/mcarenso/figures/rain_out_of_MCS/DYAMOND/'+'timelag'+str(timelag)+'/'


from myFuncs import get_MCS_contour_mask
fig = plt.figure(figsize=(14, 8))


for i_t in np.arange(args.start, args.end):
    # Assuming the following functions for loading the data
    Z = Re.load_PW(i_t)
    Precip = Re.loadPrec(i_t)
    Grey = Re.loadLabels(i_t+timelag)
    Grey_contour = get_MCS_contour_mask(Grey.values)

    plt.pcolormesh(Z, cmap='viridis', alpha=1, figure=fig)
    #plt.colorbar(label='Eau précipitable $(mm)$')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    Grey = np.where(np.isnan(Grey), 1, 0)
    plt.pcolormesh(Grey, cmap='Greys', alpha=0.25, figure=fig, label='MCS')

    plt.pcolormesh(Grey_contour, cmap='Greys', alpha=0.25, figure=fig, label='MCS')

    ## Work with masked arrays dude
    
    threshold_value = np.percentile(Precip, 99.9)
    precip_dots = np.where(Precip > threshold_value, 1, 0)
    plt.pcolormesh(precip_dots, cmap='Greys', alpha=0.25, figure=fig, label='P_99')

    # Save the plot as a PNG file
    output_path = my_path + str(i_t) + '.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
