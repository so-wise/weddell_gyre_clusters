#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load datasets and subsets
"""

def load_profile_subset(lon_min, lon_max, lat_min, lat_max, zmin, zmax): 
#load_profile_subset(lon_min, lon_max, lat_min, lat_max, zmin, zmax)
    
    # import xarray
    import xarray as xr
    import numpy as np

    # load the ctds, floats, and seals
    ctds = xr.open_mfdataset('../../../so-chic-data/CTD/*.nc', 
                             concat_dim='iPROF', combine='nested')
    floats = xr.open_mfdataset('../../../so-chic-data/FLOATS/*.nc', 
                               concat_dim='iPROF', combine='nested')
    seals = xr.open_mfdataset('../../../so-chic-data/SEALS/*.nc', 
                              concat_dim='iPROF', combine='nested')
    
    # combine into single xarray.Dataset object
    profiles = xr.combine_nested([ctds, floats, seals], concat_dim='iPROF')
    
    # assign depth coordinate
    profiles.coords['iDEPTH'] = profiles.prof_depth[0,:].values
    
    # select subset of data between 0-1000 dbar
    profiles = profiles.sel(iDEPTH=slice(zmin,zmax))
    
    # rename some of the variables
    profiles = profiles.rename({'iDEPTH':'depth',
                                'iPROF':'profile',
                                'prof_lon':'lon',
                                'prof_lat':'lat'})
    
    # drop the "prof_depth" variable, because it's redundant
    profiles = profiles.drop_vars({'prof_depth'})
    
    # change lon and lat to coordinates
    profiles = profiles.set_coords({'lon','lat'})
    
    # only keep a subset of the data variables, as we don't need them all
    profiles = profiles.get(['prof_date','prof_YYYYMMDD',
                             'prof_HHMMSS','prof_T','prof_S'])
    
    # select lat/lon section using the subsetting parameters specified above 
    profiles = profiles.where(profiles.lon<=lon_max,drop=True)
    profiles = profiles.where(profiles.lon>=lon_min,drop=True)
    profiles = profiles.where(profiles.lat<=lat_max,drop=True)
    profiles = profiles.where(profiles.lat>=lat_min,drop=True)
    
    # drop any remaining profiles with NaN values
    # the profiles with NaN values likely don't have measurements 
    # in the selected depth range
    profiles = profiles.dropna('profile')
    
    # --- handle time formatting
    
    # select MITprof values
    ntime_array_ymd = profiles.prof_YYYYMMDD.values
    #ntime_array_hms = profiles.prof_HHMMSS.values
    
    # select size
    nsize = ntime_array_ymd.size
    
    # create array of zeros
    time = np.zeros((nsize,), dtype='datetime64[s]')
    
    # loop over all values, convert do datetime64[s]
    for i in range(nsize):
        # extract strings for ymd and hms
        s_ymd = str(ntime_array_ymd[i]).zfill(8)
        # hms doesn't matter and has errors. 
        # set to noon and ignore it
        #s_hms = str(ntime_array_hms[i]).zfill(8)
        s_hms = '120000'
        # problem with 24:00:00
        #if s_hms=='240000.0':
        #    s_hms = '235959.0'
        # format into yyyy-mm-dd hh:mm:ss
        date_str_ymd = s_ymd[0:4] + '-' + s_ymd[4:6] + '-' + s_ymd[6:8]
        date_str_hms = s_hms[0:2] + ':' + s_hms[2:4] + ':' + s_hms[4:6]
        date_str =  date_str_ymd + ' ' + date_str_hms
        # convert to datetime64 (the 's' stands for seconds)
        time[i] = np.datetime64(date_str,'s')
        
    # convert to pandas datetime (may not may not end up using this)
    #time_pd = pd.to_datetime(time)
        
    # convert time array into a DataArray
    da = xr.DataArray(time, dims=['profile'])
    
    # add DataArray as new data variable to DataSet
    profiles['time'] = da
    
    # set time as a coordinate
    profiles = profiles.set_coords('time')
    
    return profiles
