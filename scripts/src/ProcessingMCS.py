from myImports import *
from PrecipGrid import PrecipGrid
from thermoFuncs import saturationSpecificHumidity

class ProcessMCS(PrecipGrid):
    def __init__(self):
        PrecipGrid.__init__(self)
        self.path_label = self.path + 'TOOCAN_SEG/'   ## goes from 20160801-i to 20161001-i with i in [0,40] 
        self.label_files = np.sort(os.listdir(self.path_label))
        
        if self.sim == 'SAM':
            self.label_files = np.sort(os.listdir(self.path_label))
            self.indexes_to_ignore = [862,  863,  958,  959, 1054, 1055, 1150, 1151, 1246, 1247, 1342, 1343, 1438, 1439, 1534, 1535, 1630, 1631, 1726, 1727, 1822, 1823]
            self.label_files = np.delete(self.label_files, self.indexes_to_ignore)
            # self.label_files = np.delete(self.label_files, len(self.label_files)-1)
            self.label_files = self.label_files[self.n_first_data_incomplete:]
            self.label_files = np.delete(self.label_files, 0) ## trying to remove first one instead of last one
    
        self.variables_func = {'PW' : self.load_PW, 'WindConv' : self.load_WindConv, 'QVsat' : self.load_QVsat}
        
        self.variables = ['PW', 'WindConv', 'QVsat']
        self.diags = ['ac', 'c', 'MCS']
             
    def loadLabels(self, i_t):
        labels = xr.open_dataarray(self.path_label + self.label_files[i_t]).load()[0]
        return labels
    
    def load_PW(self, i_t):
        root = self.df.iloc[i_t]['path_dyamond']
        if type(root) != str : print(root)
        file_precac = root+'.PW.2D.nc'
        pw = xr.open_dataarray(os.path.join(self.path_data2d, file_precac)).load()[0]
        return pw

    def load_U10m(self, i_t):
        root = self.df.iloc[i_t]['path_dyamond']
        if type(root) != str : print(root)
        file_precac = root+'.U10m.2D.nc'
        u10m = xr.open_dataarray(os.path.join(self.path_data2d, file_precac)).load()[0]
        return u10m
      
    def load_V10m(self, i_t):
        root = self.df.iloc[i_t]['path_dyamond']
        if type(root) != str : print(root)
        file_precac = root+'.V10m.2D.nc'
        v10m = xr.open_dataarray(os.path.join(self.path_data2d, file_precac)).load()[0]
        return v10m        

    def load_T2m(self, i_t):
        root = self.df.iloc[i_t]['path_dyamond']
        if type(root) != str : print(root)
        file = root+'.T2mm.2D.nc'
        T2m = xr.open_dataarray(os.path.join(self.path_data2d, file)).load()[0]
        return T2m 

    def load_PSFC(self, i_t):
        root = self.df.iloc[i_t]['path_dyamond']
        if type(root) != str : print(root)
        file = root+'.PSFC.2D.nc'
        PSFC = xr.open_dataarray(os.path.join(self.path_data2d, file)).load()[0]
        return PSFC
    
    def load_QVsat(self, i_t):
        t2m = self.load_T2m(i_t).values
        psfc = self.load_PSFC(i_t).values
        t2m_shape = t2m.shape
        psfc_shape = psfc.shape
        assert psfc.shape == t2m_shape
        qvsat = saturationSpecificHumidity(t2m.flatten(), psfc.flatten())
        qvsat = qvsat.reshape(t2m_shape)
        
        del t2m
        del psfc
        
        return qvsat    

    def load_WindConv(self, i_t):
        u = self.load_U10m(i_t)
        v = self.load_V10m(i_t)
        du_dx = np.gradient(u, axis=1)
        dv_dy = np.gradient(v, axis=0)
        wind_conv = -du_dx - dv_dy
                
        del u
        del v
        del du_dx
        del dv_dy
        gc.collect()
        return wind_conv

    def save(self, var, diag, day0 = 0, dayf = 14):
        key = var+'_'+diag
            
        try: 
            self.ds[key]
        except KeyError : 
            print(key + ' not saved, saving...')
            all_days= []
            for day_i in range(day0, dayf):
                da_day_i = self.compute_day(var, diag, day_i, key = var+'_'+diag)
                all_days.append(da_day_i)
            ## concat the list of dataarrays along days dimensions
            da_all_days = xr.concat(all_days, dim = 'days')
            ## add the dataarray to the dataset
            self.ds[key]=da_all_days
            if os.path.isfile(self.path_safeguard+'all.nc'):
                os.remove(self.path_safeguard+'all.nc')
            self.ds.to_netcdf(self.path_safeguard+'all.nc', format='NETCDF4', mode='w')
            return self.ds
        else : print(key+ ' already saved, skipping...')
        
        #close the dataset
        self.ds.close()
        
    def compute_day(self, var, diag, day, key):
        filename = key+'/day_'+str(day)+'.pkl'

        get_var = self.variables_func[var]
        
        
        if os.path.isfile(self.path_safeguard+filename):
            da_day = pickle.load(open(self.path_safeguard+filename, 'rb'))
        else:
            da_day_list = []
            for idx in self.index_per_days[day]:
                # Get the data for the day
                data = get_var(idx) # We could remove previous test for get var and apply the MCS mask here 
                if diag == 'MCS' :
                    labels = self.loadLabels(idx)
                    mask = ~np.isnan(labels.values)
                    data = np.where(mask, data, np.nan)
                    del mask 
                    del labels
                # Compute the mean precipitation
                da_day = self.spatial_max_data_from_center_to_global(data) if diag == 'c' else self.spatial_mean_data_from_center_to_global(data)
                ## first prec computed has no value
                if idx !=0 : da_day_list.append(da_day)
            stacked_mean_array = np.stack(da_day_list, axis=0)
            ## hstack the data in a dataframe of same shape
            da_day = np.max(stacked_mean_array, axis=0) if diag == 'c' else np.mean(stacked_mean_array, axis=0)
            da_day = np.expand_dims(da_day, axis=2)
            ## save as pkl file in the directory QVsat_c
            with open(self.path_safeguard+filename, 'wb') as f:
                pickle.dump(da_day, f)
                
        da_day = xr.DataArray(da_day, dims=['lat_global', 'lon_global', 'days'], coords={'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': [day]})
        
        return da_day