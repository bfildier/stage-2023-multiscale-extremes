#!/bin/bash

# Set the directory where your .nc files are located
input_dir="/bdd/MT_WORKSPACE/REMY/RCEMIP/SAM/300K/3Dfiles/"
output_dir="/home/mcarenso/code/stage-2023-multiscale-extremes/outputs/SAM_2D_extract/"

# Create a temporary directory to store intermediate output files
tmp_dir=$(mktemp -d)

# Loop through each netCDF file in the input directory
for file in ${input_dir}/*.nc
do
    # Extract the file name (without extension) from the full path
    file_name=$(basename "$file" .nc)
    
    # Construct the output file path by appending the file name to the temporary directory
    output_file="${tmp_dir}/${file_name}_2D.nc"
    
    # Use ncks to extract the 2D plan and save it to the temporary directory
    ncks -O -d z,18 "${file}" "${output_file}"
done

# Concatenate the temporary output files into a single output file, ordered by time
ncrcat -O ${tmp_dir}/*.nc "${output_dir}/Z_5km.nc"

ncatted -O -h -a history,global,o,c,"Data extracted from ${intput_dir} at Z = 5km (18th row) and concatenated with the script extract2Dplanbash" "${output_dir}/Z_5km.nc" 

# ncwa -O -a z "${output_dir}/Z_5km.nc" "${output_dir}/Z_5km.nc" doesn't work cause of RAM. Adress it in python or within the loop
# Remove the temporary directory and its contents
rm -r "${tmp_dir}"
