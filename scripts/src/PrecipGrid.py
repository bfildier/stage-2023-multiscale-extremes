from myImports import *

class PrecipGrid():
    def __init__(self, verbose = False, name = 'DYAMOND', region = '130E_165E_0N_20N', sim = 'SAM'):
        self.name = name
        self.region = region
        self.sim = sim
        self.path = '/scratchu/bfildier/'+self.name+'_REGIONS/'+self.region+'/'+self.sim+'/'
        self.path_data2d = self.path + '2D/' ## goesfrom 4608_0000000240 to 4608_0000460800
        self.data_names = ['CWP', 'LHF', 'OM500', 'OM850', 'Precac', 'PSFC', 'PW', 'RH500', 'SHF', 'T2mm', 'U10m', 'V10m']
        self.path_safeguard = '/homedata/mcarenso/Stage2023/'+self.name+'_REGIONS/'+self.region+'/'+self.sim+'/'
        self.verbose = verbose
        self.df = self.loadRelTable()
        
        # could be a function to work around the dataframe with original data
        if self.sim == 'SAM':
            self.indexes_to_ignore = [862,  863,  958,  959, 1054, 1055, 1150, 1151, 1246, 1247, 1342, 1343, 1438, 1439, 1534, 1535, 1630, 1631, 1726, 1727, 1822, 1823]
            self.df = self.df.drop(self.indexes_to_ignore).reset_index(drop=True)
            self.n_first_data_incomplete = sum(1-self.df["path_dyamond"].str.split(".").apply(lambda x : int(x[1][-7:-1]) >= 30000))
            self.df = self.df[self.df["path_dyamond"].str.split(".").apply(lambda x : int(x[1][-7:-1]) >= 30000)].reset_index(drop=True)
            self.index_per_days = [list(group.index) for _, group in self.df.groupby(['month', 'day'])]
            self.index_per_days.remove(self.index_per_days[-1])# last day empty
            self.n_days = len(self.index_per_days)
        
        #could be included in previous func
        for file in os.listdir(self.path_data2d):
            self.template_native_df = xr.open_dataset(self.path_data2d+file)
            break
        
        self.prepare_data()

        # there is a weird bug making the fill all.nc to be corrupted some times...
        # working around for now consist in deleting it and then recreating with build_xarray. 
        # Although weirdly the pixel surf and global pixel surf variables do not seem to survive this process
        # Calling save_mean_precip seems to corrupt it
        if os.path.isfile(self.path_safeguard+'all.nc'):
            self.ds = xr.open_dataset(self.path_safeguard+'all.nc')
            self.ds.close()
        else:
            self.ds  = self.build_xarray()
            self.create_day_dim()
            self.ds.to_netcdf(self.path_safeguard+'all.nc')  
            self.ds.close()

        print(self)
        print(self.ds)
        
    def saveMeanPrecip(self, day0, dayf):
        try: 
            self.ds['mean_prec']
        except KeyError : 
            print('Mean prec not saved, saving...')
            mean_prec= []
            for i in range(self.n_days):
                da_day_i = self.compute_mean_prec_by_day(i)
                mean_prec.append(da_day_i)
            ## concat the list of dataarrays along days dimensions
            da_mean_prec = xr.concat(mean_prec, dim = 'days')
            ## add the dataarray to the dataset
            self.ds['mean_prec']=da_mean_prec
            if os.path.isfile(self.path_safeguard+'all.nc'):
                os.remove(self.path_safeguard+'all.nc')
            self.ds.to_netcdf(self.path_safeguard+'all.nc', format='NETCDF4', mode='w')
        else : print('Mean prec already saved, skipping...')
        
        #close the dataset
        self.ds.close()

    def saveMaxPrecip(self, day0, dayf):
        try: 
            self.ds['max_prec']
        except KeyError : print('Max prec not saved, saving...')
        else : print('Max prec already saved, loading...')
        max_prec= []
        for i in range(self.n_days):
            da_day_i = self.compute_max_prec_by_day(i)
            max_prec.append(da_day_i)
        ## concat the list of dataarrays along days dimensions
        da_max_prec = xr.concat(max_prec, dim = 'days')
        ## add the dataarray to the dataset
        self.ds['max_prec']=da_max_prec
        if os.path.isfile(self.path_safeguard+'all.nc'):
            os.remove(self.path_safeguard+'all.nc')
        self.ds.to_netcdf(self.path_safeguard+'all.nc')
        self.ds.close()
    
    def compute_mean_prec_by_day(self, day):
        """
        Compute the mean precipitation for a given day
        """
        
        filename = 'mean_Prec/day_'+str(day+26)+'.pkl' ## +26 meanPrecip and maxprecip computed for 40 days instead of 14
        ## load it if it exists
        if os.path.isfile(self.path_safeguard+filename):
            mean_prec = pickle.load(open(self.path_safeguard+filename, 'rb'))
        else:
            day_mean_prec = []
            for idx in self.index_per_days[day]:
                # Get the data for the day
                data = self.loadPrec(idx)
                # Compute the mean precipitation
                mean_prec = self.spatial_mean_data_from_center_to_global(data)
                ## first prec computed has no value
                if idx !=0 : day_mean_prec.append(mean_prec)
            stacked_mean_array = np.stack(day_mean_prec, axis=0)
            ## hstack the data in a dataframe of same shape than
            mean_prec = np.mean(stacked_mean_array, axis=0) 
            mean_prec = np.expand_dims(mean_prec, axis=2)
            ## save as pkl file in the directory mean_Prec
            with open(self.path_safeguard+filename, 'wb') as f:
                pickle.dump(mean_prec, f)
                
        da_day = xr.DataArray(mean_prec, dims=['lat_global', 'lon_global', 'days'], coords={'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': [day]})
        
        return da_day

    def compute_max_prec_by_day(self, day):
        """
        Compute the mean precipitation for a given day
        """
        
        filename = 'max_Prec/day_'+str(day+26)+'.pkl'
        ## load it if it exists
        if os.path.isfile(self.path_safeguard+filename):
            max_prec = pickle.load(open(self.path_safeguard+filename, 'rb'))
        else:
            max_prec = []
            for idx in self.index_per_days[day]:
                # Get the data for the day
                data = self.loadPrec(idx)
                # Compute the mean precipitation
                day_max_prec = self.spatial_max_data_from_center_to_global(data)
                ## first prec computed has no value
                if idx !=0 : max_prec.append(day_max_prec)
            stacked_max_array = np.stack(max_prec, axis=0)
            ## hstack the data in a dataframe of same shape than
            max_prec = np.max(stacked_max_array, axis=0) 
            print(max_prec.shape)
            max_prec = np.expand_dims(max_prec, axis=2)
            ## save as pkl file in the directory mean_Prec
            with open(self.path_safeguard+filename, 'wb') as f:
                pickle.dump(max_prec, f)
                
        da_day = xr.DataArray(max_prec, dims=['lat_global', 'lon_global', 'days'], coords={'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': [day]})
        
        return da_day
     
    def __repr__(self):
        print('entire map', self.global_area*self.n_lon*self.n_lat)
        print('tranche de latitude : ', self.global_area*self.n_lon)
        print('tranche de longitude : ', self.global_area*self.n_lat)
        print('s', self.global_area)
        return self.name + ' ' + self.region + ' ' + self.sim
        
    def main(self):  
        for i_t in range(1, len(self.df)):
            precip = self.loadPrec(i_t, self.df)
            da_precip_i_t = xr.DataArray(precip, dims = ['lat', 'lon'], coords = {'lat': self.template_native_df['lat'].values, 'lon': self.template_native_df['lon'].values})

    def prepare_data(self):
        self.lat_centers, self.lon_centers = self.template_native_df['lat'].values, self.template_native_df['lon'].values
        self.lat_length_on_center, self.lon_length_on_center = self.__compute_length_centers_from_coord_borders__()
        
        self.pixel_surface = self.lat_length_on_center * self.lon_length_on_center
        
        self.n_lat, self.n_lon = 20, 32 ##choose your grid size here         

        self.global_area = self.pixel_surface.sum()/self.n_lat/self.n_lon #depending on the remeshed grid point surface you want computed here
        self.global_lat_area = self.global_area*self.n_lon 
        
        self.lat_global = [i for i in range(self.n_lat)]
        self.lon_global = [j for j in range(self.n_lon)]
        
        ## We start by computing the area of each latitude band
        self.lat_area = np.sum(self.pixel_surface, axis=1)
        self.cumsum_lat_area = np.cumsum(self.lat_area)
        
        self.i_min, self.i_max, self.alpha_i_min, self.alpha_i_max = self.__get_i_and_alpha_lat__()

        self.slices_i_lat = [slice(i_min+1, i_max) for i_min, i_max in zip(self.i_min[:,0], self.i_max[:,0])]        
        self.area_by_lon_and_global_lat = self._compute_area_by_lon_()
        self.cumsum_area_by_lon_and_global_lat = np.cumsum(self.area_by_lon_and_global_lat, axis = 1)
        
        self.j_min, self.j_max, self.alpha_j_min, self.alpha_j_max = self.__get_j_and_alpha_lon__()
        self.j_min, self.j_max =self.j_min.astype(int), self.j_max.astype(int)

        self.slices_j_lon = self.__build_slices_j_lon__()
        
        self.grid_surface = self.sum_data_from_center_to_global(self.pixel_surface)
                        
    def create_day_dim(self):
        self.n_days = len(self.index_per_days)
        if type(self.n_days) is not int:
            raise ValueError('number of days is not an integer')
        self.days_dim = self.ds.expand_dims('days', None)
        self.days_values = np.arange(self.n_days)
        self.ds['days'] = xr.DataArray(np.arange(self.n_days), dims = ['days']) 

    def build_xarray(self):
        
        da = xr.DataArray(self.pixel_surface, dims = ['lat', 'lon'], coords = {'lat': self.lat_centers, 'lon': self.lon_centers})
        
        self.global_pixel_surface = self.sum_data_from_center_to_global(self.pixel_surface)

        da_global = xr.DataArray(self.global_pixel_surface, dims = ['lat_global', 'lon_global'], coords = {'lat_global': self.lat_global, 'lon_global': self.lon_global})
        
        ds = xr.Dataset({'pixel_surf': da, 'global_pixel_surf': da_global})
                
        return ds

    def __get_i_and_alpha_lat__(self):
        i_min, i_max = np.zeros((self.n_lat, self.n_lon)), np.zeros((self.n_lat, self.n_lon))
        alpha_i_min, alpha_i_max = np.ones((self.n_lat, self.n_lon)), np.ones((self.n_lat, self.n_lon))

        for i_lat, cum_length in enumerate(self.cumsum_lat_area):
            border_left = cum_length - self.lat_area[i_lat]
            border_right = cum_length
            
            for i in range(self.n_lat):
                cum_global_length = (i+1)*self.global_lat_area
                
                if cum_global_length > border_left and ((cum_global_length < border_right) or (math.isclose(cum_global_length, border_right))):
                    bottom_contrib = (cum_global_length - border_left)/self.lat_area[i_lat]
                    top_contrib = (border_right - cum_global_length)/self.lat_area[i_lat]           
                    if self.verbose : print('local', i_lat, cum_length,'global',  i, cum_global_length,'borders',  border_left, border_right, 'contribs', bottom_contrib, top_contrib)
                    if i != self.n_lat-1:
                        i_min[i+1, :] = i_lat
                        alpha_i_min[i+1, :] = top_contrib if not (math.isclose(cum_global_length, border_right)) else 0
                    

                    i_max[i, :] = i_lat
                    alpha_i_max[i, :] = bottom_contrib if not (math.isclose(cum_global_length, border_right)) else 1
                    
        return i_min.astype(int), i_max.astype(int), alpha_i_min, alpha_i_max

    def __get_j_and_alpha_lon__(self):
        j_min, j_max = np.zeros((self.n_lat, self.n_lon)), np.zeros((self.n_lat, self.n_lon))
        alpha_j_min, alpha_j_max = np.ones((self.n_lat, self.n_lon)), np.ones((self.n_lat, self.n_lon))
        for i in range(self.n_lat):
            cumsum_area_by_lon = self.cumsum_area_by_lon_and_global_lat[i, :]
            for j_lon, cum_length in enumerate(cumsum_area_by_lon):
                border_left = cum_length - self.area_by_lon_and_global_lat[i, j_lon]
                border_right = cum_length
                
                for j in range(self.n_lon):
                    cum_global_length = (j+1)*self.global_area
                    
                    if cum_global_length > border_left  and ((cum_global_length) < border_right or (math.isclose(cum_global_length, border_right))):

                        left_contrib = (cum_global_length - border_left)/self.area_by_lon_and_global_lat[i, j_lon]
                        right_contrib = (border_right - cum_global_length)/self.area_by_lon_and_global_lat[i, j_lon]
                        if self.verbose : print('local', j_lon, cum_length,'global',  j, cum_global_length,'borders',  border_left, border_right, 'contribs', left_contrib, right_contrib)
                        if j!= self.n_lon-1:
                            j_min[i, j+1] = j_lon
                            alpha_j_min[i, j+1] = right_contrib if not (math.isclose(cum_global_length, border_right)) else 0
                            
                        j_max[i, j] = j_lon
                        alpha_j_max[i, j] = left_contrib if not (math.isclose(cum_global_length, border_right)) else 1
    
        return j_min, j_max, alpha_j_min, alpha_j_max    
     
    def __build_slices_j_lon__(self):
        slices_j_lon = np.empty((self.n_lat, self.n_lon), dtype=object)
        for i in range(self.n_lat):
            for j in range(self.n_lon):
                slices_j_lon[i, j] = slice(int(self.j_min[i, j])+1, int(self.j_max[i, j])) 
        return slices_j_lon
    
    def _compute_area_by_lon_(self):
            area_by_lon = np.zeros((self.n_lat, self.lon_centers.shape[0]))
            for j_lon in range(self.lon_centers.shape[0]):
                for i, slice_i_lat in enumerate(self.slices_i_lat):
                    i_min = self.i_min[i, :]
                    i_min = self.check_all_values_same(i_min)
                    i_max = self.i_max[i, :]
                    i_max = self.check_all_values_same(i_max)
                    alpha_i_min = self.alpha_i_min[i, :]
                    alpha_i_min = self.check_all_values_same(alpha_i_min)
                    alpha_i_max = self.alpha_i_max[i, :]
                    alpha_i_max = self.check_all_values_same(alpha_i_max)
                
                    ## print i_min, i_max, alpha_i_min, alpha_i_max
                    if self.verbose : print(i, i_min, i_max, alpha_i_min, alpha_i_max)
                    bottom_sum = self.pixel_surface[i_min,j_lon]*alpha_i_min
                    if self.verbose : print(bottom_sum)
                    top_sum = self.pixel_surface[i_max,j_lon]*alpha_i_max
                    if self.verbose : print(top_sum)
                    mid_sum = np.sum(self.pixel_surface[slice_i_lat, j_lon])
                    if self.verbose : print(mid_sum)
                    area_by_lon[i, j_lon] = mid_sum+bottom_sum+top_sum
                    #print everything 
                    if False : print('i', i, 'j_lon', j_lon, 'i_min', i_min, 'i_max', i_max, 'slice_i_lat', slice_i_lat, 'alpha_i_min', alpha_i_min, 'alpha_i_max', alpha_i_max, 'bottom_sum', bottom_sum, 'top_sum', top_sum, 'mid_sum', mid_sum, 'area_by_lon', area_by_lon[i, j_lon])
            
            return area_by_lon
        
    def sum_data_from_center_to_global(self, data_on_center):
        x = data_on_center
        X = np.zeros((self.n_lat, self.n_lon))
        for i, slice_i_lat in enumerate(self.slices_i_lat):
            for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                if self.verbose : print(slice_i_lat, slice_j_lon)
                mid_sum = np.sum(x[slice_i_lat, slice_j_lon])
                bottom_sum = np.sum( x[self.i_min[i,j], slice_j_lon]*self.alpha_i_min[i,j])
                top_sum = np.sum( x[self.i_max[i,j], slice_j_lon]*self.alpha_i_max[i,j])
                left_sum = np.sum( x[slice_i_lat, self.j_min[i,j]]*self.alpha_j_min[i,j])
                right_sum = np.sum( x[slice_i_lat, self.j_max[i,j]]*self.alpha_j_max[i,j])
                bottom_left_corner = x[self.i_min[i,j], self.j_min[i,j]]*self.alpha_j_min[i,j]*self.alpha_i_min[i,j]
                bottom_right_corner = x[self.i_min[i,j], self.j_max[i,j]]*self.alpha_j_max[i,j]*self.alpha_i_min[i,j]
                top_left_corner = x[self.i_max[i,j], self.j_min[i,j]]*self.alpha_j_min[i,j]*self.alpha_i_max[i,j]
                top_right_corner = x[self.i_max[i,j], self.j_max[i,j]]*self.alpha_j_max[i,j]*self.alpha_i_max[i,j]
                X[i, j] = mid_sum+bottom_sum+top_sum+left_sum+right_sum+bottom_left_corner+bottom_right_corner+top_left_corner+top_right_corner
        return X
    
    def spatial_mean_data_from_center_to_global(self, data_on_center):
        x = data_on_center*self.pixel_surface if type(data_on_center) == np.ndarray else data_on_center.values*self.pixel_surface
        X = np.zeros((self.n_lat, self.n_lon))
        for i, slice_i_lat in enumerate(self.slices_i_lat):
            for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                if self.verbose : print(slice_i_lat, slice_j_lon)
                mid= x[slice_i_lat, slice_j_lon].flatten()
                bottom = x[self.i_min[i,j], slice_j_lon]*self.alpha_i_min[i,j].flatten()
                top = x[self.i_max[i,j], slice_j_lon]*self.alpha_i_max[i,j].flatten()
                left = x[slice_i_lat, self.j_min[i,j]]*self.alpha_j_min[i,j].flatten()
                right = x[slice_i_lat, self.j_max[i,j]]*self.alpha_j_max[i,j].flatten()
                bottom_left_corner = x[self.i_min[i,j], self.j_min[i,j]]*self.alpha_j_min[i,j]*self.alpha_i_min[i,j]
                bottom_right_corner = x[self.i_min[i,j], self.j_max[i,j]]*self.alpha_j_max[i,j]*self.alpha_i_min[i,j]
                top_left_corner = x[self.i_max[i,j], self.j_min[i,j]]*self.alpha_j_min[i,j]*self.alpha_i_max[i,j]
                top_right_corner = x[self.i_max[i,j], self.j_max[i,j]]*self.alpha_j_max[i,j]*self.alpha_i_max[i,j]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    X[i, j] = np.nanmean(np.concatenate([mid ,bottom ,top ,left ,right ,
                                            np.array([bottom_left_corner ,bottom_right_corner ,top_left_corner ,top_right_corner])]))

        return X/self.grid_surface       
    
    def spatial_max_data_from_center_to_global(self, data_on_center):
        x = data_on_center if type(data_on_center) == np.ndarray else data_on_center.values
        X = np.zeros((self.n_lat, self.n_lon))
        alpha_max = self.__build_alpha_max__()
        for i, slice_i_lat in enumerate(self.slices_i_lat):
            for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                if self.verbose : print(slice_i_lat, slice_j_lon)
                m = np.nanmax(x[slice_i_lat, slice_j_lon].flatten())
                b = np.nanmax((x[self.i_min[i,j], slice_j_lon]*alpha_max[self.i_min[i,j], slice_j_lon]).flatten())
                t = np.nanmax((x[self.i_max[i,j], slice_j_lon]*self.alpha_max[self.i_max[i,j], slice_j_lon]).flatten())
                l = np.nanmax((x[slice_i_lat, self.j_min[i,j]]*self.alpha_max[slice_i_lat, self.j_min[i,j]]).flatten())
                r = np.nanmax((x[slice_i_lat, self.j_max[i,j]]*self.alpha_max[slice_i_lat, self.j_max[i,j]]).flatten())
                blc = (x[self.i_min[i,j], self.j_min[i,j]]*self.alpha_max[self.i_min[i,j], self.j_min[i,j]]).flatten()
                btc = (x[self.i_min[i,j], self.j_max[i,j]]*self.alpha_max[self.i_min[i,j], self.j_max[i,j]]).flatten()
                tlc = (x[self.i_max[i,j], self.j_min[i,j]]*self.alpha_max[self.i_max[i,j], self.j_min[i,j]]).flatten()
                trc = (x[self.i_max[i,j], self.j_max[i,j]]*self.alpha_max[self.i_max[i,j], self.j_max[i,j]]).flatten()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    X[i, j] = np.nanmax(np.array([m, b, t, l, r, blc, btc, tlc, trc]))
        print(X.shape, X)
        return X     
                
    def check_all_values_same(self, arr):
        first_value = arr[0]
        for value in arr:
            if value != first_value:
                raise ValueError("Array contains different values")

        return first_value  # No error raised if all values are the same

    def loadPrec(self, i_t):
        # Load DYAMOND-SAM Precac

        precac_prev = self.loadPrecac(i_t-1,self.df)
        precac_current = self.loadPrecac(i_t,self.df)
        
        # get 30mn precipitation from difference
        prec = precac_current - precac_prev
        prec.rename('precipitation')
        
        # free up memory
        del precac_prev
        del precac_current
        gc.collect()
    
        return prec     

    def loadPrecac(self, i_t, df):
        root = df.iloc[i_t]['path_dyamond']
        if type(root) != str : print(root)
        file_precac = root+'.Precac.2D.nc'
        # load
        precac = xr.open_dataarray(os.path.join(self.path_data2d, file_precac)).load()[0]
        return precac
    
    def loadRelTable(self, which='DYAMOND_SEG'):
        
        # relation table DYAMOND-SAM -- TOOCAN segmentation masks
        if which == 'DYAMOND_SEG':
            df = pd.read_csv('/home/mcarenso/code/stage-2023-multiscale-extremes/input/relation_2_table_UTC_dyamond_segmentation.csv')
            df.sort_values(by='UTC',ignore_index=True,inplace=True)
        return df   
    
    def __get_coord_border_from_centers__(self, coord_centers):
        coord_borders = list()
        coord_borders.append(np.floor(coord_centers[0]))
        for i in range(len(coord_centers)-1):
            coord_borders.append((coord_centers[i]+coord_centers[i+1])/2)
        coord_borders.append(np.ceil(coord_centers[-1]))  
        return coord_borders

    def __compute_length_centers_from_coord_borders__(self):
        lat_length = np.zeros(shape=(len(self.lat_centers), len(self.lon_centers)))
        lon_length = np.zeros(shape=(len(self.lat_centers), len(self.lon_centers)))
        
        self.lat_borders = self.__get_coord_border_from_centers__(self.lat_centers)
        self.lon_borders = self.__get_coord_border_from_centers__(self.lon_centers)
        
        for i_lat in range(len(self.lat_borders)-1):
            for j_lon in range(len(self.lon_borders)-1):
                lat1, lat2, lon1, lon2 = self.lat_borders[i_lat], self.lat_borders[i_lat+1], self.lon_borders[j_lon], self.lon_borders[j_lon+1]
                lat_length[i_lat, j_lon] = self.haversine(lat1, lon1, lat2, lon1)
                lon_length[i_lat, j_lon] = self.haversine(lat1, lon1, lat1, lon2)
        return lat_length, lon_length

    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the distance between two points on the Earth (specified in decimal degrees)
        using the Haversine formula.
        """
        R = 6371  # Radius of the Earth in kilometers

        # Convert decimal degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c

        return distance
          
    def imshow(self, x):
        plt.figure(figsize=(20,10))
        print(x.shape)
        plt.imshow(x, origin = 'lower')
        plt.colorbar()

    def __build_alpha_max__(self):
        self.alpha_max = np.zeros(shape=(len(self.lat_centers), len(self.lon_centers)))
        for i, slice_i_lat in enumerate(self.slices_i_lat):
            for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                self.alpha_max[slice_i_lat, slice_j_lon] = 1
                self.alpha_max[self.i_min[i,j], slice_j_lon] = 1 if self.alpha_i_min[i,j] > 0.5 else 0
                self.alpha_max[self.i_max[i,j], slice_j_lon] = 1 if self.alpha_i_max[i,j] > 0.5 else 0
                self.alpha_max[slice_i_lat, self.j_min[i,j]] = 1 if self.alpha_j_min[i,j] > 0.5 else 0
                self.alpha_max[slice_i_lat, self.j_max[i,j]] = 1 if self.alpha_j_max[i,j] > 0.5 else 0
                
                self.alpha_max[self.i_min[i,j], self.j_min[i,j]] = 1 if self.alpha_i_min[i,j] > 0.5 and self.alpha_j_min[i,j] > 0.5 else 0
                self.alpha_max[self.i_max[i,j], self.j_min[i,j]] = 1 if self.alpha_i_max[i,j] > 0.5 and self.alpha_j_min[i,j] > 0.5 else 0
                self.alpha_max[self.i_min[i,j], self.j_max[i,j]] = 1 if self.alpha_i_min[i,j] > 0.5 and self.alpha_j_max[i,j] > 0.5 else 0
                self.alpha_max[self.i_max[i,j], self.j_max[i,j]] = 1 if self.alpha_i_max[i,j] > 0.5 and self.alpha_j_max[i,j] > 0.5 else 0
                
        return self.alpha_max