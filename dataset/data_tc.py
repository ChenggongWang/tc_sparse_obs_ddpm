#!/usr/bin/env python
# Wenchang Yang (wenchang@princeton.edu)
# Thu Oct  1 16:36:42 EDT 2020 
import sys, os.path, os, glob
import xarray as xr, numpy as np, pandas as pd 
from numpy import pi as Pi, cos, sin
import cftime 
import typer
from tqdm import tqdm
def main(dataname:str, year:int, negative_natural:bool=False):
    '''
    dataname: variable name
    year: year in simulation: /tigress/wenchang/MODEL_OUT/HIRAM/CTL1990s_v201910_tigercpu_intelmpi_18_540PE/POSTP. From 101-200
    negative_natural: rotate local (x,y) coordinate to negative natural coordinate (TC moves to the x-negative direction). Default: False

    ''' 
    
    if dataname == 'precip':
        data_name = 'precip'
        scale_factor = 24*3600
        units = 'mm/day'
    elif dataname == 'slp':
        data_name = 'slp'
        scale_factor = 1
        units = 'mb'
    elif dataname == 'v_ref':
        data_name = 'v_ref'
        scale_factor = 1
        units = 'm/s'
    elif dataname == 'u_ref':
        data_name = 'u_ref'
        scale_factor = 1
        units = 'm/s'
    elif dataname == 'WVP':
        data_name = 'WVP'
        scale_factor = 1
        units = 'kg/m2'
    else:
        raise Exception(f"Please define scale_fator and units for {dataname}")

    odata_name = f'{data_name}_xy'

    R = 6370 #km, earth radius
    L1 = 2*Pi*R/360 #km, distance of one degree longitude at the equator (or latitude at any location)
    # x = xr.DataArray(np.arange(-500, 501, 10), dims='x', attrs=dict(units='km'))
    # y = xr.DataArray(np.arange(-500, 501, 10), dims='y', attrs=dict(units='km'))
    x = xr.DataArray(np.arange(-790, 791, 20), dims='x', attrs=dict(units='km'))
    y = xr.DataArray(np.arange(-790, 791, 20), dims='y', attrs=dict(units='km'))
    if negative_natural:
        x = x.assign_attrs(long_name='negative natural s')
        y = y.assign_attrs(long_name='negative natural n')

    # composite variable
    ifile = f'/tigress/wenchang/MODEL_OUT/HIRAM/CTL1990s_v201910_tigercpu_intelmpi_18_540PE/POSTP/{year:04d}0101.atmos_4xdaily.nc'
    da = xr.open_dataset(ifile)[data_name] \
        .pipe(lambda x: x*scale_factor) \
        .assign_attrs(units=units) \
        .rename(grid_xt='lon', grid_yt='lat')
    # tc tracks
    ds = xr.open_dataset('HIRAM.CTL1990s_v201910_tigercpu_intelmpi_18_540PE.tc_tracks.TS.0101-0200.nc') \
        .sel(year=year)

    # time axis adjustment
    # change year so that time range is 2000-01-01 00:00:00 to 2001-01-01 00:00:00 
    time = da.time
    time_new = [cftime.DatetimeGregorian(yr-year+2000, month, day, hour)
        for yr,month,day,hour in zip(time.dt.year.values, time.dt.month.values, time.dt.day.values, time.dt.hour.values)]
    da = da.assign_coords(time=time_new)
    # no values on Dec 31: shift time index one day backward from Mar 1st to the end of year
    if not ds.hour.where((ds.month==12)&(ds.day==31)).max()>0: 
        time = da.time
        times = np.array([f'{yr:04d}-{month:02d}-{day:02d} {hour:02d}:00:00'
            for yr,month,day,hour in zip(time.dt.year.values, time.dt.month.values, time.dt.day.values, time.dt.hour.values)])
        time_new = da.indexes['time'].where( times<'2000-03-01 00:00:00', other=da.indexes['time'].shift(-1, 'D') )
        da = da.assign_coords(time=time_new)
    da = da.assign_coords(time=da.indexes['time'].shift(-1, 'H')) # so that no 24:00:00 (not recognized by python)

    # interpolation loop
    zz = np.zeros((ds.storm.size, ds.stage.size, y.size, x.size)) + np.nan
    angle_tcmotion = np.zeros((ds.storm.size, ds.stage.size)) + np.nan
    vtcmotion = np.zeros((ds.storm.size, ds.stage.size)) + np.nan
    for istorm in tqdm(range(ds.storm.size), desc=f"{data_name} #tc "):
        #print('istorm:', istorm)
        if np.isnan(ds.isel(storm=istorm, stage=0).lon.item()):
            # exit if all TCs are beeing processed (#tc is less than 200)
            break
        for istage in range(ds.stage.size):
            # print('istorm:', istorm, istage)      
            ds0 = ds.isel(storm=istorm, stage=istage)
            lon0 = ds0.lon.item()
            if np.isnan(lon0):
                break
            lat0 = ds0.lat.item()
            month0 = ds0.month.item()
            day0 = ds0.day.item()
            hour0 = ds0.hour.item() - 1 # minus 1 to be consistent with da.time

            # track point time stamp
            t0 = f'2000-{month0:02d}-{day0:02d}T{hour0:02d}'

            # local coordinates
            # rotate local coordinate so that TC moves to the left (direction of negative x axis): negative natural coordinate
            if negative_natural:
                # get dlat and dlon
                if istage == 0: # first track point: forward difference
                    ds_next = ds.isel(storm=istorm, stage=1)
                    dlat = ds_next.lat.item() - ds0.lat.item()
                    dlon = ds_next.lon.item() - ds0.lon.item()
                elif np.isnan(ds.isel(storm=istorm, stage=istage+1).lon.item()): # last track point: backward difference
                    ds_prev = ds.isel(storm=istorm, stage=istage-1)
                    dlat = ds0.lat.item() - ds_prev.lat.item()
                    dlon = ds0.lon.item() - ds_prev.lon.item()
                else: #central difference
                    ds_next = ds.isel(storm=istorm, stage=istage+1)
                    ds_prev = ds.isel(storm=istorm, stage=istage-1)
                    dlat = ( ds_next.lat.item() - ds_prev.lat.item() )/2
                    dlon = ( ds_next.lon.item() - ds_prev.lon.item() )/2
                angle_tcmotion[istorm, istage] = np.angle(dlon*cos(lat0*Pi/180) + dlat*1j, deg=True)# angle (in degree) between TC motion and zonal direction
                dx, dy, dt = L1*dlon*cos(lat0*Pi/180)*1000, L1*dlat*1000, 6*3600 # in units of m, m, s
                vtcmotion[istorm, istage] = (dx**2 + dy**2)**0.5/dt # tc motion speed: m/s
                theta = np.angle(-dlon*cos(lat0*Pi/180) - dlat*1j) # angle between x-negative in rotating axis and zonal direction
                # x-negative natural coordinate (x,y) projected onto local (x_, y_) coordinate
                x_ = x*cos(theta) - y*sin(theta)
                y_ = x*sin(theta) + y*cos(theta)
                lat_xy = (y_+x_)*0 + lat0 + y_/L1 
                lon_xy = (y_+x_)*0 + lon0 + x_/(L1*cos(lat_xy*Pi/180))
            else:
                lat_xy = (y+x)*0 + lat0 + y/L1 
                lon_xy = (y+x)*0 + lon0 + x/(L1*cos(lat_xy*Pi/180))

            da0 = da.sel(time=t0).squeeze() \
                .interp(lon=lon_xy, lat=lat_xy) \
                .drop(['lon', 'lat', 'time'])
            da0 = da0.transpose('y', 'x')

            zz[istorm, istage, :, :] = da0.values

    da_ = xr.DataArray(zz, dims=['storm', 'stage', 'y', 'x'], 
        coords=[ds.storm, ds.stage, y, x], attrs=dict(units=units))

    ds[odata_name] = da_
    if negative_natural:
        angle_tcmotion = xr.DataArray(angle_tcmotion, dims=['storm', 'stage'], 
            coords=[ds.storm, ds.stage], 
            attrs=dict(units='deg', long_name='angle between TC motion and zonal east'))
        ds['angle_tcmotion'] = angle_tcmotion
        vtcmotion = xr.DataArray(vtcmotion, dims=['storm', 'stage'],
            coords=[ds.storm, ds.stage],
            attrs=dict(units='m/s', long_name='TC motion speed'))
        ds['vtcmotion'] = vtcmotion
    output_name = f'./tc_data/tc_{data_name}.{year}.nc'
    ds.to_netcdf(
        output_name,
        encoding={odata_name: {'zlib': True, 'complevel': 1, 'dtype': 'float32'}}
    )
    print(f"output saved to {output_name}")

if __name__ == '__main__':
    typer.run(main) 
    
