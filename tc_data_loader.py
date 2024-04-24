import xarray as xr
import numpy as np
import torch 
from torch.utils.data import Dataset  

class TC_xy_Dataset(Dataset):
    def __init__(self, data_dir:str, data_vars:list, years:list, transform=None, target_transform=None): 
        self.lat_lim = 35
        self.data_dir = data_dir
        self.data_vars = data_vars
        self.years = years 
        self.channels = len(data_vars)
        self.data = self._get_data_all() 
        self.length = self.data.shape[0]
        assert self.data.shape[-1]==self.data.shape[-2]
        self.image_size = self.data.shape[-1]
        self.transform = transform
        self.target_transform = target_transform

        self._normalization()
        
    def _get_data_all(self):
        # load data into cpu memory
        data_all = [] 
        for ci, var in enumerate(self.data_vars):
            icount = 0
            data_sv = []
            for year in self.years: 
                with xr.open_dataset(f'{self.data_dir}/tc_{var}.{year}.nc').load() as data_ds:
                    data_sy_sv, length_sy = self._get_data_one_da(data_ds, f'{var}_xy') 
                    print(f'var {var} year {year}: length {length_sy} ') 
                    icount = icount + length_sy 
                    data_sv.append(data_sy_sv)
            data_sv = torch.cat(data_sv,0)[:,None,]
            data_all.append(data_sv)
        data_all = torch.cat(data_all,1)
        return data_all 
        
    def _get_data_one_da(self, ds_tc_data, var):
        # count available tc snapshots in one ds file
        length_one_file = 0
        data = []
        for istorm in range(ds_tc_data.storm.size):
            #print('istorm:', istorm)   
            ds_s = ds_tc_data.isel(storm=istorm)
            lat_s = ds_s.lat
            if np.isnan(lat_s.isel(stage=0)):
                # exit if all TCs are beeing processed (#tc is less than 200)
                break
            for istage in range(lat_s.stage.size):
                lat_ss = lat_s.isel(stage=istage).data
                if np.isnan(lat_ss):
                    break
                if lat_ss <  self.lat_lim  and lat_ss > - self.lat_lim :
                    if not np.isnan(ds_s[var].isel(stage=istage).data).sum(): 
                        data.append(ds_s[var].isel(stage=istage).data) 
                        length_one_file = length_one_file + 1
        return torch.tensor(np.array(data)), length_one_file
    
    def __len__(self): 
        return self.length

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data
        
    def unnormalization(self, data):
        data_unnorm = data * self.scale + self.offset
        return data_unnorm
        
    def _normalization(self):
        # [-1, 1] 
        self.offset = self.data.numpy().mean(axis=(0))[None,:]
        self.scale = self.data.numpy().std(axis=(0))[None,:]*2
        self.offset = self.offset
        self.data = (self.data-self.offset)/self.scale 
        return