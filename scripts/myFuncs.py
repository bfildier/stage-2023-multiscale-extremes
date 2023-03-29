import xarray as xr

def explore_nc(file_path, var = None):
    # Load the NetCDF file using xarray
    dataset = xr.open_dataset(file_path)

    # Print the contents of the dataset
    print(dataset)

    # Print the dimensions of the dataset
    print('Dimensions: ', dataset.dims)

    # Print the variables in the dataset
    print('Variables: ', list(dataset.variables))

    # Print the coordinates in the dataset
    print('Coordinates: ', list(dataset.coords))

    # Print the attributes of the dataset
    print('Attributes: ', dataset.attrs)

    # Access a specific variable and print its attributes
    if var != None :
        var = dataset['your_variable']
        print(var)
        print('Attributes: ', var.attrs)

    # Close the dataset
    dataset.close()