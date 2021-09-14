#####################################################################
# Utilities for loading profile data, slicing
#####################################################################

# import pakcages
import numpy as np
import xarray as xr
from sklearn import preprocessing
from sklearn.decomposition import PCA
import umap
import random

#####################################################################
# Load the profile data (combined CTD, float, and seal data)
#####################################################################
def load_profile_data(data_location, lon_min, lon_max,
                      lat_min, lat_max, zmin, zmax, zscale=False):

    # start message
    print('load_and_preprocess.load_profile_data')

    # load the ctds, floats, and seals
    ctds = xr.open_mfdataset(data_location + 'CTD/*.nc',
                             concat_dim='iPROF', combine='nested')
    floats = xr.open_mfdataset(data_location + 'FLOATS/*.nc',
                               concat_dim='iPROF', combine='nested')
    seals = xr.open_mfdataset(data_location + 'SEALS/*.nc',
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
    profiles = profiles.get(['prof_date','prof_YYYYMMDD','prof_HHMMSS','prof_T','prof_S'])

    # either use the z-scaling (no discarded profiles) or use geometric bounds
    # for discarding profiles
    if zscale==True:
        print('This feature is not ready yet *******************'')
        profiles = z_scaling(profiles)
    else:
        profiles = profiles.where(profiles.lon<=lon_max,drop=True)
        profiles = profiles.where(profiles.lon>=lon_min,drop=True)
        profiles = profiles.where(profiles.lat<=lat_max,drop=True)
        profiles = profiles.where(profiles.lat>=lat_min,drop=True)
        # drop any remaining profiles with NaN values
        # the profiles with NaN values don't have measurements in selected depth range
        profiles = profiles.dropna('profile')

    # start message
    print('----> profiles loaded')

    # return
    return profiles

#####################################################################
# z-scaling
#####################################################################
def z_scaling(df):
    print('This feature is not ready yet ***')
    return df

#####################################################################
# Handle date and time data
#####################################################################
def preprocess_time_and_date(profiles):

    # start message
    print('load_and_preprocess.preprocess_time_and_data')

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

    # examine Dataset again
    return profiles

#####################################################################
# Apply preprocessing scaling
#####################################################################
def apply_scaling(profiles):

    # start message
    print('load_and_preprocess.apply_scaling')

    # scale salinity
    XS = profiles.prof_SA
    scaled_S = preprocessing.scale(XS)
    scaled_S.shape

    # scale temperature
    XT = profiles.prof_CT
    scaled_T = preprocessing.scale(XT)
    scaled_T.shape

    # concatenate
    Xraw = np.concatenate((XT,XS),axis=1)
    Xscaled = np.concatenate((scaled_T,scaled_S),axis=1)

    return Xraw, Xscaled

#####################################################################
# Fit and apply PCA (applied to absolute salinity, conservative temp)
#####################################################################
def fit_and_apply_pca(profiles, number_of_pca_components=3):

    # start message
    print('load_and_preprocess.fit_and_apply_pca')

    # concatenate
    Xraw, Xscaled = apply_scaling(profiles)

    # create PCA object
    pca = PCA(number_of_pca_components)

    # fit PCA model
    pca.fit(Xscaled)

    # transform input data into PCA representation
    Xpca = pca.transform(Xscaled)

    # add PCA values to the profiles Dataset
    #PCA1 = xr.DataArray(Xpca[:,0],dims='profile')
    #PCA2 = xr.DataArray(Xpca[:,1],dims='profile')
    #PCA3 = xr.DataArray(Xpca[:,2],dims='profile')

    # calculated total variance explained
    total_variance_explained_ = np.sum(pca.explained_variance_ratio_)
    print(total_variance_explained_)

    return pca, Xpca

#####################################################################
# Fit and apply UMAP
#####################################################################
def fit_and_apply_umap(ploc,profiles,n_neighbors=50,min_dist=0.0,frac=0.33):

    # start message
    print('load_and_preprocess.fit_and_apply_umap')

    # apply scaling
    Xraw, Xscaled = apply_scaling(profiles)

    # random sample
    rsample_size = int(frac*Xscaled.shape[0])
    rows_id = random.sample(range(0,Xscaled.shape[0]-1), rsample_size)
    Xscaled_for_umap = Xscaled[rows_id,:]

    # fit UMAP to scaled data
    embedding = umap.UMAP(n_neighbors=n_neighbors,
                          min_dist=min_dist,
                          n_components=3,
                          random_state=42).fit(Xscaled_for_umap)

    # transform
    Xumap = embedding.transform(Xscaled)

    return embedding, Xumap
