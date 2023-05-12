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

def count_rel_nan(arr):
    output = None
    if len(arr.flatten()) == 0 : output = np.nan
    else : output = np.count_nonzero(np.isnan(arr))/len(arr.flatten())
    return output

def idx_by_label(labels, label_list):
    idxs = [label_list.index(label) for label in labels]
    return idxs
    
def condFracPrec(arr, axis = None, threshold = 0):
    num_over_threshold = np.count_nonzero(arr > threshold, axis=axis)/(48*32*32)
    return num_over_threshold

def interpolate_nan_values(Z):
    # Find the indices of the NaN values in the input matrix
    nan_indices = np.argwhere(np.isnan(Z))

    # Generate the coordinate arrays for the non-NaN data
    non_nan_indices = np.argwhere(~np.isnan(Z))

    # Get the non-NaN values from the input matrix
    non_nan_values = Z[~np.isnan(Z)]

    # Use griddata to interpolate the non-NaN values at the NaN indices
    interp_data = griddata(non_nan_indices, non_nan_values, nan_indices, method='linear')

    # Use LinearNDInterpolator to perform a more accurate interpolation
    interp_func = LinearNDInterpolator(non_nan_indices, non_nan_values)
    interp_data[np.isnan(interp_data)] = interp_func(nan_indices[np.isnan(interp_data)])

    # Replace the NaN values in Z with the interpolated values
    Z[np.isnan(Z)] = interp_data

    return Z

def fit_interPrec(ListData, func, nd=4):
    len_bins = 10*nd +2

    a_values = np.zeros_like(ListData[1])
    cc_values = np.zeros_like(ListData[1])
    redchi_values = np.zeros_like(ListData[1])


    y_values = []
    t_values = np.array([295, 300, 305])
    for i in range(len_bins):
        for j in range(len_bins):
            y_values = np.log([ListData[0][i,j], ListData[1][i,j], ListData[2][i,j]])
            if np.isnan(y_values).any() or np.isinf(y_values).any():
                # skip fitting if any value is NaN
                a_values[i,j] = np.nan
                cc_values[i,j] = np.nan
                redchi_values[i,j] = np.nan
            else:
                # Define the lmfit model
                model = Model(func)
                params = model.make_params(temp=300, a = -20, cc=7.0)

                # Perform the fit
                result = model.fit(y_values, params, temp=t_values)

                # Extract the fit results
                a_values[i,j] = result.params['a'].value
                cc_values[i,j] = 100*result.params['cc'].value
                redchi_values[i,j] = result.redchi
    return(cc_values, redchi_values)

def calculate_first_derivative(time_series, time_step = 30):
    """
    Calculate the first derivative of a time series using the central difference method.

    Parameters:
    time_series (np.array): A 1D numpy array representing the surface of an object over time.
    time_step (float): The time interval between successive samples in the time series.

    Returns:
    derivative (np.array): A 1D numpy array representing the first derivative of the time series.
    """
    n = len(time_series)
    derivative = np.zeros(n)

    # Calculate the central difference method
    for i in range(1, n-1):
        derivative[i] = (time_series[i+1] - time_series[i-1]) / (2*time_step)

    # Apply forward difference at the start and backward difference at the end
    derivative[0] = (time_series[1] - time_series[0]) / time_step
    derivative[n-1] = (time_series[n-1] - time_series[n-2]) / time_step

    # Robustness check for stability
    derivative[np.isnan(derivative)] = 0.0
    derivative[np.isinf(derivative)] = 0.0

    return derivative

def classify_derivative(derivative):
    """
    Classify a derivative array based on its positiveness.

    Parameters:
    derivative (np.array): A 1D numpy array representing the derivative of a time series.

    Returns:
    classification (int): A flag indicating the classification of the derivative.
    """
    # Step 1: Calculate the array of positiveness flags
    positiveness = np.zeros_like(derivative)
    positiveness[derivative >= 0] = 1  # positive derivative
    positiveness[derivative < 0] = -1  # negative derivative

    # Step 2: Classify the derivative based on its positiveness and length
    classification = 0
    length = len(derivative)
    positive_periods = []
    negative_periods = []

    if length < 10:
        # Case 1: Length is lower than 10
        classification = 1

    else:
        # Case 2: Length is 10 or greater
        if positiveness[0] > 0:
            # Start of a positive period
            positive_periods.append(0)
        for i in range(1, length):
            if positiveness[i] > 0 and positiveness[i-1] <= 0:
                # Start of a positive period
                positive_periods.append(i)
            elif positiveness[i] < 0 and positiveness[i-1] >= 0:
                # Start of a negative period
                negative_periods.append(i)

        if len(positive_periods) > 1 and len(negative_periods) != 0:
            # First period is negative
            classification = 2.1
        else :
            classification = 2

    return classification, positive_periods, negative_periods

def growthRate(arr, axis=None, MCS_list = None):
    from load_TOOCAN_DYAMOND_modif_BenAndMax import load_TOOCAN_DYAMOND
    file_tracking='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2022_04/irtb/FileTracking/TOOCAN-SAM_RCE_large300_2D_irtb.dat.gz'
    MCS_list = load_TOOCAN_DYAMOND(file_tracking)

    label_list = [MCS_list[i].label for i in range(len(MCS_list))]
    ## Get the labels of MCS for these (x,y,t)s, for now remove Nans and don't discriminate MCS that may occur multiple time.
    new_shape = tuple(np.delete(np.array(arr.shape), axis))
    flattened = np.reshape(arr, new_shape + (-1,))
    days,lat,lon = flattened.shape[:3]
    output = np.zeros((days,lat,lon))
    for t in range(days) : 
        for y in range(lat) :
            for x in range(lon) :
                labels = np.unique(flattened[t,y,x,:], axis=-1)

                labels_no_nan = [x for x in labels if not math.isnan(x)]
                MCSs = [MCS_list[idx] for idx in idx_by_label(labels_no_nan, label_list)] ## while this idx correspond to the indexing of MCS within the tracking file.
                assert len(labels_no_nan)==len(MCSs)
                MCS_growthRateMean = []
                for i in range(len(MCSs)):
                    surf = np.array(MCSs[i].clusters.surf235k_km2)
                    dsurf = calculate_first_derivative(surf)
                    MClass, pos, neg = classify_derivative(dsurf)
                    if (MClass==2) or MClass == 1 :
                        MCS_growthRateMean.append(np.nanmean(dsurf[dsurf>0]))
                        #if (dsurf>0).any() == False : print(MCSs[i].label) #16221 17756 22397
                    elif MClass==2.1:
                        MCS_growthRateMean.append(np.nanmean(dsurf[pos[0]:neg[0]]))
                if len(MCS_growthRateMean)>0 : 
                    output[t,y,x] = np.nanmean(MCS_growthRateMean)
    return output
    
def maskMean(arr, axis=None, **kwargs):
    
    #TOOCAN segmentation masks (les labels des objets MCS, dans la grille originale x,y,t)
    temp = "300"
    file_seg='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2022_04/irtb/TOOCAN_2.07_SAM_RCE_large'+temp+'_2D_irtb.nc'   ## TODO PPass arguemnt temp correctly
    n_days = 20
    label_mask = xr.open_dataarray(file_seg).isel(time=slice(48*n_days)) ## TODO Pass arguemnt n_days correctly
    mask = ~np.isnan(label_mask.values)
    mask = mask[ :, :,:, np.newaxis, np.newaxis, np.newaxis]
    mask = np.reshape(mask, arr.shape)
    arr_masked = arr * mask
    output = np.mean(arr_masked, axis = axis) 
    print(np.all(arr_masked == arr))
        
    return output
    
def maskMax(arr, axis=None, **kwargs):
    
    #TOOCAN segmentation masks (les labels des objets MCS, dans la grille originale x,y,t)
    temp = "300"
    file_seg='/bdd/MT_WORKSPACE/MCS/RCE/SAM/TOOCAN/TOOCAN_v2022_04/irtb/TOOCAN_2.07_SAM_RCE_large'+temp+'_2D_irtb.nc'   ## TODO PPass arguemnt temp correctly
    n_days = 20
    label_mask = xr.open_dataarray(file_seg).isel(time=slice(48*n_days)) ## TODO Pass arguemnt n_days correctly
    mask = ~np.isnan(label_mask.values)
    mask = mask[ :, :,:, np.newaxis, np.newaxis, np.newaxis]
    mask = np.reshape(mask, arr.shape)
    arr_masked = arr * mask
    output = np.max(arr_masked, axis = axis) 
        
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