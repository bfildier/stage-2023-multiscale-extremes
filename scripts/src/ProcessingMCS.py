from myImports import *
from PrecipGrid import PrecipGrid

class ProcessMCS(PrecipGrid):
    def __init__(self):
        PrecipGrid.__init__(self)
        self.path_label = self.path + 'TOOCAN_SEG/'   ## goes from 20160801-i to 20161001-i with i in [0,40] 
        self.label_files = np.sort(os.listdir(self.path_label))
        
        if self.sim == 'SAM':
            self.label_files = np.sort(os.listdir(self.path_label))
            self.indexes_to_ignore = [862,  863,  958,  959, 1054, 1055, 1150, 1151, 1246, 1247, 1342, 1343, 1438, 1439, 1534, 1535, 1630, 1631, 1726, 1727, 1822, 1823]
            self.label_files = np.delete(self.label_files, self.indexes_to_ignore)
            self.label_files = np.delete(self.label_files, len(self.label_files)-1)
            self.label_files = self.label_files[self.n_first_data_incomplete:]
        
    
        self.diagnostics_func = {'PW_MCS': self.compute_PW_MCS, 'WindConv_MCS' : self.compute_WindConv_MCS}
                     
    def saveMean(self, key, day0 = 0, dayf = 14):
        get_da_day_i = self.diagnostics_func[key]
        try: 
            self.ds[key]
        except KeyError : 
            print('Mean prec not saved, saving...')
            mean_prec= []
            for i in range(day0, dayf):
                da_day_i = get_da_day_i(i)
                mean_prec.append(da_day_i)
            ## concat the list of dataarrays along days dimensions
            da_mean_prec = xr.concat(mean_prec, dim = 'days')
            ## add the dataarray to the dataset
            self.ds[key]=da_mean_prec
            if os.path.isfile(self.path_safeguard+'all.nc'):
                os.remove(self.path_safeguard+'all.nc')
            self.ds.to_netcdf(self.path_safeguard+'all.nc', format='NETCDF4', mode='w')
            return self.ds
        else : print(key+ 'already saved, skipping...')
        
        #close the dataset
        self.ds.close()
             
    def loadLabels(self, i_t):
        labels = xr.open_dataarray(self.path_label + self.label_files[i_t]).load()[0]
        return labels
    
    def loadPW(self, i_t):
        root = self.df.iloc[i_t]['path_dyamond']
        if type(root) != str : print(root)
        file_precac = root+'.PW.2D.nc'
        pw = xr.open_dataarray(os.path.join(self.path_data2d, file_precac)).load()[0]
        return pw
    
    def load_PW_by_MCS(self, i_t):
        pw = self.loadPW(i_t)
        labels = self.loadLabels(i_t)
        mask = ~np.isnan(labels.values)
        pw_mcs = pw * mask
        
        del pw
        del mask
        del labels 
        gc.collect()
        return pw_mcs

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
        
    def load_WindConv_by_MCS(self, i_t):
        u = self.load_U10m(i_t)
        v = self.load_V10m(i_t)
        du_dx = np.gradient(u, axis=1)
        dv_dy = np.gradient(v, axis=0)
        wind_conv = du_dx + dv_dy
        labels = self.loadLabels(i_t)
        mask = ~np.isnan(labels.values)
        
        WindConv_by_MCS = wind_conv * mask
        
        del u
        del v
        del du_dx
        del dv_dy
        del wind_conv
        del mask
        del labels 
        gc.collect()
        return WindConv_by_MCS

    def compute_WindConv_MCS(self, day):
            filename = 'WindConv_MCS/day_'+str(day)+'.pkl'
            ## load it if it exists
            if os.path.isfile(self.path_safeguard+filename):
                mean_prec = pickle.load(open(self.path_safeguard+filename, 'rb'))
            else:
                day_mean = []
                for idx in self.index_per_days[day]:
                    # Get the data for the day
                    data = self.load_WindConv_by_MCS(idx)
                    # Compute the mean precipitation
                    mean_prec = self.spatial_mean_data_from_center_to_global(data)
                    ## first prec computed has no value
                    if idx !=0 : day_mean.append(mean_prec)
                stacked_mean_array = np.stack(day_mean, axis=0)
                ## hstack the data in a dataframe of same shape
                mean_prec = np.mean(stacked_mean_array, axis=0) 
                mean_prec = np.expand_dims(mean_prec, axis=2)
                ## save as pkl file in the directory PW_MCS
                with open(self.path_safeguard+filename, 'wb') as f:
                    pickle.dump(mean_prec, f)
                    
            da_day = xr.DataArray(mean_prec, dims=['lat_global', 'lon_global', 'days'], coords={'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': [day]})
            
            return da_day
 
    def compute_PW_MCS(self, day):
        filename = 'PW_MCS/day_'+str(day)+'.pkl'
        ## load it if it exists
        if os.path.isfile(self.path_safeguard+filename):
            mean_prec = pickle.load(open(self.path_safeguard+filename, 'rb'))
        else:
            day_mean = []
            for idx in self.index_per_days[day]:
                # Get the data for the day
                data = self.load_PW_by_MCS(idx)
                # Compute the mean precipitation
                mean_prec = self.spatial_mean_data_from_center_to_global(data)
                ## first prec computed has no value
                if idx !=0 : day_mean.append(mean_prec)
            stacked_mean_array = np.stack(day_mean, axis=0)
            ## hstack the data in a dataframe of same shape
            mean_prec = np.mean(stacked_mean_array, axis=0) 
            mean_prec = np.expand_dims(mean_prec, axis=2)
            ## save as pkl file in the directory PW_MCS
            with open(self.path_safeguard+filename, 'wb') as f:
                pickle.dump(mean_prec, f)
                
        da_day = xr.DataArray(mean_prec, dims=['lat_global', 'lon_global', 'days'], coords={'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': [day]})
        
        return da_day