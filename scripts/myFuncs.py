from myImports import *

def explore_nc(file_path, var = None): ## Worst than print(xarray.ds)
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

def plot_nc_files_1var_overTime_from_dir(dir_path, var_name):

    # Get a list of all .nc files in the directory
    file_list = [f for f in os.listdir(dir_path) if f.endswith('.nc')]

    # Loop over the files and load them into xarray datasets
    datasets = []
    for file in file_list:
        file_path = os.path.join(dir_path, file)
        ds = xr.open_dataset(file_path)
        datasets.append(ds)

    # Plot the variable over the time dimension for each dataset in subplots
    num_plots = len(datasets)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 5*num_plots))

    for i, ds in enumerate(datasets):
        var_name = "DAILYmcs_Pop"  # replace with the name of the variable you want to plot
        ds[var_name].mean(axis=2).plot.line(x='time', ax=axs[i])
        axs[i].set_title(f"{file_list[i]}")
        axs[i].set_xlabel("time")
        axs[i].set_ylabel(var_name)

    plt.tight_layout()
    plt.show()    

