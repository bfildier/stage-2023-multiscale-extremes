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

class Normalize_IL:
    def __init__(self, nd):
        self.nd = nd

    def _transform_IL(self, x):
        x = x/100
        return  np.clip((-np.log10(1-x)/self.nd), 0, 1)
    
    def _inverse_IL(self, y):
        #return (norm_data*nd)
        return 100*(1-10**-(y*self.nd))

def Ponderate_MCS_by_surface(data, MCS_surface=None, grid_surface = None, weight = False, rel_weight = False, ExtensiveVariable = False):
    """
    MCS_surface is to be given the "INT_surfMCS" variable from CACATOES and data typically has same dimension than data. Actually there could be a discusion about whever
    I didnt think of using the surfmax ones because I didn't want their max surface over all timesteps, but more their current surface.
    grid_surface is to be used to norm the ponderation over the grid
    Instead of giving both its surface and the grid surface to compute its relative surface within the gird 
    we could use other "INT_surfFraction_235K". We'll depend on the temperature chosen, but this way it seems like
    we'd consider only the surface of the MCS within the considered grid. Altough INT_MCS already seems to consider surface per X,Y

    """
    if not weight and not rel_weight:
        if ExtensiveVariable : 
            output = np.sum(data, axis = 1)
        else : 
            rel_MCS_data_mean_nan = np.nanmean(np.where(data==0, np.nan, data), axis=1) ## average with nan mean, after replacing 0 values by nan. 
            output = np.where(np.isnan(rel_MCS_data_mean_nan), 0, rel_MCS_data_mean_nan) ## Replace the computed nan by 0s. 

    elif weight and not rel_weight :
        assert MCS_surface is not None , "You must give the surface of each MCS if you want to multiply the MCS_data by the MCS_surface"
        MCS_data_surface = np.multiply(data, MCS_surface) 
        if ExtensiveVariable : 
            output = np.sum(MCS_data_surface, axis = 1)
        else : 
            rel_MCS_data_mean_nan = np.nanmean(np.where(MCS_data_surface==0, np.nan, MCS_data_surface), axis=1) ## average with nan mean, after replacing 0 values by nan. 
            output = np.where(np.isnan(rel_MCS_data_mean_nan), 0, rel_MCS_data_mean_nan) ## Replace the computed nan by 0s.

    elif rel_weight : 
        assert(MCS_surface is not None and grid_surface is not None)
        rel_MCS_surface = MCS_surface / grid_surface[:, np.newaxis, : , :]
        rel_MCS_data = np.multiply(data, rel_MCS_surface)
        if  ExtensiveVariable :
            output = np.sum(rel_MCS_data, axis =1) 
        else :
            rel_MCS_data_mean_nan = np.nanmean(np.where(rel_MCS_data==0, np.nan, rel_MCS_data), axis=1)
            output = np.where(np.isnan(rel_MCS_data_mean_nan), 0, rel_MCS_data_mean_nan)
    return output

# def Aggregate_by_MCS(data , labels, func= np.mean, weights = None, return_MCS_labels = False, test = False):
#     ### TODO : issue is that by aggregating once per label we have to choose on which x,y grid the MCS must be accounted for. 
#     """
#     Aggregates a 4D data array, typically (time, MCS, Y, X), by computing the mean of the corresponding values
#     for each unique label in a 4D label array.
    
#     Args:
#         data (numpy.ndarray): A 4D numpy array containing the data to aggregate.
#         labels (numpy.ndarray): A 4D numpy array containing the labels that
#             correspond to the data. Usually the variable "QCmcs_label" of CACATOES
#         weights (numpy.ndarray, optional): A 4D numpy array containing the
#             weights to use for the aggregation. If None, all values will be
#             weighted equally. Phylosophically we thought of the INT_surfmaxkm2_235K variables. 
#         test bool is to be given specifically the "QCmcs_label" as labels to verify that te aggreagtion 
#         works well compared to "DAILYmcs_PopMCS_Pop"
#     Returns:
#         numpy.ndarray: A 4D numpy array containing the aggregated data.
#         numpy.ndarray: A 4D numpy array containing the unique labels that were
#             used for the aggregation.
#     """
#     if test : 
#         def count_unique(x, axis):
#             return len(np.unique(x)) 
#         data = labels

#     # Flatten the arrays to make them easier to work with
#     #flat_data = data.reshape(-1)
#     #flat_labels = labels.reshape(-1)
#     #if weights is not None:
#     #    flat_weights = weights.reshape(-1)


#     # Identify the unique labels
#     unique_labels, index = np.unique(labels, axis=1, return_index=True)

#     # Compute the mean of the corresponding values in the data array
#     aggregated_data = []
#     for label in unique_labels:
#         mask = np.all(flat_labels == label, axis=1)
#         if weights is None:
#             weighted_data = flat_data[mask]
#         else:
#             weighted_data = flat_data[mask] * flat_weights[mask][:, np.newaxis]
#         mean_data = func(weighted_data, axis=1)
#         aggregated_data.append(mean_data)

#     # Reshape the aggregated data array back to the original shape of the data array
#     aggregated_data = np.array(aggregated_data).reshape(data.shape[:-1] + (-1,))

#     # Identify the unique labels in the original 4D label array
#     unique_labels_4d = np.unique(labels, axis=0)

#     # Check which unique labels were used to compute the mean
#     used_labels = unique_labels_4d[np.isin(unique_labels_4d, unique_labels, assume_unique=True).all(axis=1)]

#     return aggregated_data, used_labels
