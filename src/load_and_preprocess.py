#####################################################################
# Utilities for loading profile data, slicing
#####################################################################

# import pakcages
import numpy as np
import xarray as xr
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn import manifold
from xgcm import Grid
import random
#import umap

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

    # add variable to indicate data source
    ctds['source'] = 'ctd'
    floats['source'] = 'argo'
    seals['source'] = 'seal'

    # combine into single xarray.Dataset object
    profiles = xr.combine_nested([ctds, floats, seals],
                                 concat_dim='iPROF')

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
    profiles = profiles.get(['prof_date','prof_YYYYMMDD','prof_HHMMSS',
                             'prof_T','prof_S','source'])

    # either use the z-scaling (no discarded profiles) or use geometric bounds
    # for discarding profiles
    if zscale==True:
        print('This feature is not ready yet *******************')
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
# Load Ekman velocity (estimates from tau and oss)
#####################################################################
def load_ekman_vel(floc="../../so-chic-data/Stress_and_EKE/"):

    # start message
    print('load_and_preprocess.load_ekman_vel')

    # open dataset
    ds = xr.open_dataset(floc+"Ekmanvel.nc")

    # sort by Longitude
    ds = ds.sortby("longitude")

    # return
    return ds

#####################################################################
# Load Ekman velocity (estimates from tau and oss)
#####################################################################
def load_eke_and_geo(floc="../../so-chic-data/Stress_and_EKE/"):

    # start message
    print('load_and_preprocess.load_eke_and_geo')

    # open dataset
    ds = xr.open_dataset(floc+"geovel_eke_climatology.nc")

    # return
    return ds

#####################################################################
# Load Reeve climatology
#####################################################################
def load_reeve_climatology(floc="../../so-chic-data/WeddellGyre_OM/"):

    # start message
    print('load_and_preprocess.load_reeve_climatology')

    # load all three periods
    ds1 = xr.open_dataset(floc+"WeddellGyre_OM_Period2001to2005.nc")
    ds2 = xr.open_dataset(floc+"WeddellGyre_OM_Period2006to2009.nc")
    ds3 = xr.open_dataset(floc+"WeddellGyre_OM_Period2010to2013.nc")

    # concatenate
    ds = xr.concat((ds1,ds2,ds3), dim="time_period")

    # do some renaming
    ds = ds.rename({'Pressure Levels (dbar)':'plevs',
                    'Conservative Temperature (DegC)':'prof_CT',
                    'Absolute Salinity (gPERkg)':'prof_SA',
                    'RHO (kgPERm3)':'rho',
                    'Latitude (DegN)':'lat',
                    'Longitude (DegE)':'lon',
                    'Conservative Temperature mask based on Period2002to2013':'CTmask',
                    'Absolute Salinity mask based on Period2002to2013':'SAmask'})

    # lat, lon, pressure coordinates
    pressure = ds.plevs.values[0,:]
    lat = ds.lat.values[0,:]
    lon = ds.lon.values[0,:]
    ds = ds.drop({'plevs','lat','lon'})

    # assign coordinates
    ds = ds.assign_coords({"level":pressure, "lat":lat, "lon":lon})

    # stack
    ds = ds.stack(profile=("time_period","lat","lon"))

    # drop some variables
    ds = ds.drop_vars({'Tmax Conservative Temperature (DegC)',
            'Tmax Absolute Salinity (gPERkg)',
            'Tmax Pressure (dbar)',
            'Tmax RHO (kgPERm3)',
            'Tmax Conservative Temperature mapping error (DegC)',
            'Tmax Conservative Temperature mask based on Period of file',
            'Tmax Absolute Salinity mask based on Period of file',
            'Conservative Temperature mask based on Period of file',
            'Absolute Salinity mask based on Period of file',
            'Tmax Absolute Salinity mapping error (gPERkg)',
            'Tmax Pressure mapping error (dbar)',
            'Tmax RHO mapping error (kgPERm3)',
            'Absolute Salinity mapping error (gPERkg)',
            'RHO mapping error (kgPERm3)',
            'Conservative Temperature mapping error (DegC)',
            'Tmax Conservative Temperature mask based on Period2002to2013',
            'Tmax Absolute Salinity mask based on Period2002to2013'})

    # alter some attributes
    ds.attrs['TimeRange']='N/A'

    return ds

#####################################################################
# Load single class from previously classififed data
#####################################################################
def load_single_class(data_location, selected_class):

    # start message
    print('load_and_preprocess.load_single_class')

    # load the class
    profiles = xr.open_dataset(data_location)

    # select the single class, drop old class info
    inClass = (profiles.label == selected_class)
    profiles = profiles.isel(profile=inClass)
    profiles = profiles.drop_vars({'label','posteriors'})
    profiles = profiles.drop('CLASS')

    # start message
    print('----> single-class profiles loaded')

    # return
    return profiles

####################################################################
# z-scaling (scale profile by length of water column)
#####################################################################
def z_scaling(df):

    #df.apply_ufunc(z1Dinterp, exclude_dims=set(("depth",))

    #return df
    return

#####################################################################
# 1D interpolation (of a single vector)
#####################################################################
def z1Dinterp(x):

    # find index of first nan cell
    nan_index = np.where(np.isnan(x))[0][0]

    # just select part of vector before the upper nan (bathymetry)
    y = x[:nan_index]

    # extract depth values above the not-nan value
    z = df.depth[:nan_index].values

    # scale depth between [0,1]
    zscaled = (z-z[0])/max(z-z[0])

    # now we would have to interpolate onto a standard set of levels
    # ranging from [0,1]. Perhaps 50 levels?

    return zscaled

#####################################################################
# Handle date and time data
#####################################################################
def preprocess_time_and_date(profiles):

    # start message
    print('load_and_preprocess.preprocess_time_and_date')

    # select MITprof values
    ntime_array_ymd = profiles.prof_YYYYMMDD.values
    #ntime_array_hms = profiles.prof_HHMMSS.values

    # select size
    nsize = ntime_array_ymd.size

    # create array of zeros
    time =  np.zeros((nsize,), dtype='datetime64[s]')
    month = np.zeros((nsize,), dtype='int')
    year =  np.zeros((nsize,), dtype='int')
    season = np.zeros((nsize,), dtype='int')

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
        # grab the year and month, put them into place (integers)
        year[i] = int(s_ymd[0:4])
        month[i] = int(s_ymd[4:6])

    # convert to pandas datetime (may not may not end up using this)
    #time_pd = pd.to_datetime(time)
    
    # assign season based on the month (1=DJF, 2=MAM, 3=JJA, 4=SON)
    for i in range(nsize):
        if (month[i]==12 or month[i]==1 or month[i]==2):
            season[i] = 0
        elif (month[i]==3 or month[i]==4 or month[i]==5):
            season[i] = 1
        elif (month[i]==6 or month[i]==7 or month[i]==8):
            season[i] = 2
        elif (month[i]==9 or month[i]==10 or month[i]==11):
            season[i] = 3
        else:
            season[i] = None

    # convert time array into a DataArray, add to Dataset
    da = xr.DataArray(time, dims=['profile'])
    profiles['time'] = da

    # do the same for year, month, and season
    dy = xr.DataArray(year, dims=['profile'])
    profiles['year'] = dy
    dm = xr.DataArray(month, dims=['profile'])
    profiles['month'] = dm
    dseason = xr.DataArray(season, dims=['profile'])
    profiles['season'] = dseason

    # set time, year, month as coordinates
    profiles = profiles.set_coords({'time','year','month','season'})

    # examine Dataset again
    return profiles

#####################################################################
# Regrid onto higher-resolution vertical grid
#####################################################################
def regrid_onto_more_vertical_levels(profiles, zmin, zmax, zlevs=50):

    print('load_and_preprocess.regrid_onto_more_vertical_levels')

    # define grid object
    grid = Grid(profiles, coords={'Z': {'center': 'depth'}}, periodic=False)
    target_z_levels = np.linspace(zmin, zmax, zlevs)

    # linearly interpolate temperature onto selected z levels
    ct_on_highz = grid.transform(profiles.prof_CT, 'Z',
                                 target_z_levels,
                                 target_data=profiles.depth,
                                 method='linear')

    # linearly interpolate salt onto selected z levels
    sa_on_highz = grid.transform(profiles.prof_SA, 'Z',
                                 target_z_levels,
                                 target_data=profiles.depth,
                                 method='linear')
    
    # linearly interpolate density onto selected z levels
    sig0_on_highz = grid.transform(profiles.sig0, 'Z',
                                   target_z_levels,
                                   target_data=profiles.depth,
                                   method='linear')

    # rename dimension to avoid conflict with existing dimension
    profiles['ct_on_highz'] = ct_on_highz.rename({'depth':'depth_highz'})
    profiles['sa_on_highz'] = sa_on_highz.rename({'depth':'depth_highz'})
    profiles['sig0_on_highz'] = sig0_on_highz.rename({'depth':'depth_highz'})

    # drop any levels where the interpolation failed
    profiles = profiles.dropna(dim='depth_highz', how='all')

    return profiles

######################################################################################
# Regrid onto density levels (tends to get better results after high-z interpolation)
######################################################################################
def regrid_onto_density_levels(profiles, target_sig0_levels=None):

    print('load_and_preprocess.regrid_onto_density_levels')

    # if none provided, define target sigma levels
    if (target_sig0_levels is None):
        sig0min = profiles.sig0_on_highz.values.min()
        sig0max = profiles.sig0_on_highz.values.max()
        target_sig0_levels = np.linspace(sig0min, sig0max, 100)

    # define grid object
    grid = Grid(profiles, coords={'Z': {'center': 'depth_highz'}}, periodic=False)

    # linearly interpolate temperature onto selected z levels
    ct_on_sig0 = grid.transform(profiles.ct_on_highz, 'Z',
                                target_sig0_levels,
                                target_data=profiles.sig0_on_highz,
                                method='linear')

    # linearly interpolate salt onto selected z levels
    sa_on_sig0 = grid.transform(profiles.sa_on_highz, 'Z',
                                target_sig0_levels,
                                target_data=profiles.sig0_on_highz,
                                method='linear')
    
    # find the depth of density surfaces
    tmp1 = xr.broadcast(profiles.depth_highz,profiles.profile)
    broadcast_depth_highz = tmp1[0]
    z_on_sig0 = grid.transform(broadcast_depth_highz, 'Z',
                               target_sig0_levels,
                               target_data=profiles.sig0_on_highz,
                               method='linear')

    # rename dimension to avoid conflict with existing dimension
    profiles['ct_on_sig0'] = ct_on_sig0.rename({'sig0_on_highz':'sig0_levs'})
    profiles['sa_on_sig0'] = sa_on_sig0.rename({'sig0_on_highz':'sig0_levs'})
    profiles['z_on_sig0'] = z_on_sig0.rename({'sig0_on_highz':'sig0_levs'})

    # drop any levels where there are no values (all NaNs)
    #profiles = profiles.dropna(dim='sig0_levs', how='all')

    return profiles

#####################################################################
# Select more specific density range; drop NaNs
#####################################################################
def select_sig0_range(profiles,sig0range=(26.5,27.2)):

    print('load_and_preprocess.select_sig0_range')

    # select profiles in the more specific density range
    profiles = profiles.sel(sig0_levs=slice(sig0range[0],sig0range[1]))

    # might be redundant, but get rid of levels where all values nan
    profiles = profiles.dropna(dim='')

    # drop all profiles with nan values
    profiles = profiles.dropna(dim='profile', how='any')

    return profiles

#####################################################################
# Apply preprocessing scaling
#####################################################################
def apply_scaling(profiles, method='onZ', whichVars='both'):

    # start message
    print('load_and_preprocess.apply_scaling')

    # select SA on pressure levels or SA on sig0
    if method=='onZ':
        print('load_and_preprocess.apply_scaling: using depth levels')
        XS = profiles.prof_SA
        XT = profiles.prof_CT
    elif method=='onSig':
        print('load_and_preprocess.apply_scaling: using density levels')
        XS = profiles.sa_on_sig0
        XT = profiles.ct_on_sig0
    else:
        print('method must be onZ or onSig')

    # scale salinity and temperature
    scaled_S = preprocessing.scale(XS)
    scaled_T = preprocessing.scale(XT)

    # concatenate
    if whichVars=='both':
        Xraw = np.concatenate((XT,XS),axis=1)
        Xscaled = np.concatenate((scaled_T,scaled_S),axis=1)
    elif whichVars=='justSalt':
        Xraw = XS
        Xscaled = scaled_S
    elif whichVars=='justTemp':
        Xraw = XT
        Xscaled = scaled_T
    else:
        print("load_and_preprocess.apply_scaling, input parameter \
                whichVar must be justSalt, justTemp, or both")

    return Xraw, Xscaled

############################################################################
# Generate area-uniform training dataset
############################################################################
def select_area_uniform_training_dataset(profiles, 
                                         lon_min = -65,
                                         lon_max =  80,
                                         lat_min = -80,
                                         lat_max = -45,
                                         grid_spacing = 10.0,
                                         max_number_of_samples_from_a_cell = 500,
                                         lat_ref = -45):

    # start message
    print('load_and_preprocess.select_area_uniform_training_dataset')
    
    # single-profile Dataset that will be merged with others to form the training Dataset
    training_dataset = profiles.isel(profile=[0,1])

    # define the grid spacing and the lat/lon values of the sampling grid
    i_sampling_grid = np.arange(lon_min, lon_max, grid_spacing)
    j_sampling_grid = np.arange(lat_min, lat_max, grid_spacing)

    # loop through each cell in the sampling grid
    for i in i_sampling_grid:
        for j in j_sampling_grid:

            # define the lat-lon edges of the sampling grid cell
            is_in_cell_lon = ( profiles.lon > i ) & ( profiles.lon <= i + grid_spacing )
            is_in_cell_lat = ( profiles.lat > j ) & ( profiles.lat <= j + grid_spacing )

            # find all the profiles that are in the selected sampling grid cell
            profiles_in_cell_lon = profiles.where(is_in_cell_lon, drop=True)
            profiles_in_cell_latlon = profiles_in_cell_lon.where(is_in_cell_lat, drop=True)

            # maximum number of profiles that can be sampled in the selected sampling grid cell
            max_number_of_profiles = profiles_in_cell_latlon.profile.size

            # scale the max number of samples from a cell by the cosine of latitude
            lat_scaling = np.cos(j*np.pi/180 - lat_ref)
            max_number_of_samples_from_a_cell_scaled = round(max_number_of_samples_from_a_cell*lat_scaling)

            # select random subset of the profiles in the sampling grid cell
            if max_number_of_profiles>0:
                # the number of samples should be all the profiles, or the max number, whichever is smaller
                number_of_samples = np.minimum(max_number_of_profiles, max_number_of_samples_from_a_cell_scaled)
                # next, find the indices for the random samples
                indices_for_random_samples = random.sample(range(0,max_number_of_profiles), number_of_samples)
                drawn_from_sampling_cell = profiles_in_cell_latlon.isel(profile=indices_for_random_samples)

                # merge the profiles drawn from the sampling cell with the rest of the training dataset
                training_dataset = training_dataset.merge(drawn_from_sampling_cell)\
                
    return training_dataset

############################################################################
# Fit and apply PCA (applied to absolute salinity, conservative temp)
############################################################################
def fit_and_apply_pca(profiles, number_of_pca_components=3, kernel=False, 
                      train_frac=0.33, method='onZ', whichVars='both'):

    # start message
    print('load_and_preprocess.fit_and_apply_pca')

    # concatenate
    Xraw, Xscaled = apply_scaling(profiles, method, whichVars)

    # create PCA object
    if kernel==True:
        # KernelPCA approach (crashses due to memory)
        print('load_and_preprocess: apply KernelPCA')
        pca = KernelPCA(n_components=number_of_pca_components,
                        kernel='linear', fit_inverse_transform=True, gamma=10)
    else:
        pca = PCA(number_of_pca_components)

    # random sample for training
    pf = profiles.profile
    rsample_size = np.min((int(train_frac*pf.size),int(pf.size)))
    rows_id = random.sample(range(0,pf.size), rsample_size)
    Xtrain = Xscaled[rows_id,:]

    # fit PCA model using training dataset
    print('Fitting PCA')
    pca.fit(Xtrain)

    # transform entire input dataset into PCA representation
    Xpca = pca.transform(Xscaled)

    # calculated total variance explained
    if kernel==False:
        total_variance_explained_ = np.sum(pca.explained_variance_ratio_)
        print(total_variance_explained_)

    return pca, Xpca

############################################################################
# Fit PCA
############################################################################
def fit_pca(training_dataset, number_of_pca_components=3, kernel=False, 
            train_frac=0.99, method='onZ', whichVars='both'):

    # start message
    print('load_and_preprocess.fit_pca')

    # concatenate
    Xraw, Xscaled = apply_scaling(training_dataset, method, whichVars)

    # create PCA object
    if kernel==True:
        # KernelPCA approach (crashses due to memory)
        print('load_and_preprocess: apply KernelPCA')
        pca = KernelPCA(n_components=number_of_pca_components,
                        kernel='linear', fit_inverse_transform=True, gamma=10)
    else:
        pca = PCA(number_of_pca_components)

    # random sample for training
    pf = training_dataset.profile
    rsample_size = np.min((int(train_frac*pf.size),int(pf.size)))
    rows_id = random.sample(range(0,pf.size), rsample_size)
    Xtrain = Xscaled[rows_id,:]

    # fit PCA model using training dataset
    print('Fitting PCA')
    pca.fit(Xtrain)

    return pca

#####################################################################
# Apply an existing PCA
#####################################################################
def apply_pca(profiles, pca, method='onZ', whichVars='both'):

    # start message
    print('load_and_preprocess.apply_pca')

    # concatenate
    Xraw, Xscaled = apply_scaling(profiles, method, whichVars)

    # transform
    Xpca = pca.transform(Xscaled)

    # calculated total variance explained
    total_variance_explained_ = np.sum(pca.explained_variance_ratio_)
    print(total_variance_explained_)

    return Xpca

#####################################################################
# Fit and apply t-SNE
#####################################################################
def fit_and_apply_tsne(profiles, Xpca, random_state=0, perplexity=50,
                       tsne_frac=0.10, var_to_plot="label"):

    # sample size
    sample_size = np.min((int(tsne_frac*Xpca.shape[0]),int(Xpca.shape[0])))

    # random sample for tSNE plot
    rows_id = random.sample(range(0,Xpca.shape[0]), sample_size)
    Xpca_for_tSNE = Xpca[rows_id,:]
    
    # select which variable to plot
    if var_to_plot=="label":
        colors_for_tSNE = profiles.label[rows_id].values
    elif var_to_plot=="Tmin":
        colors_for_tSNE = profiles.Tmin[rows_id].values
    elif var_to_plot=="Tmax":
        colors_for_tSNE = profiles.Tmax[rows_id].values
    elif var_to_plot=="Smin":
        colors_for_tSNE = profiles.Smin[rows_id].values
    elif var_to_plot=="Smax":
        colors_for_tSNE = profiles.Smax[rows_id].values
    elif var_to_plot=="SA":
        colors_for_tSNE = profiles.prof_SA[rows_id].values
    elif var_to_plot=="CT":
        colors_for_tSNE = profiles.prof_CT[rows_id].values
    elif var_to_plot=="sig0":
        colors_for_tNSE = profiles.sig0[rows_id].values
    elif var_to_plot=="dyn_height":
        colors_for_tSNE = profiles.dyn_height[rows_id].values
    elif var_to_plot=="mld":
        colors_for_tSNE = profiles.mld[rows_id].values
    elif var_to_plot=="imetric":
        colors_for_tSNE = profiles.imetric[rows_id].values
    else:
        colors_for_tSNE = profiles.label[rows_id].values

    # create tSNE object
    tsne = manifold.TSNE(n_components=2, init='random',
                         random_state=random_state,
                         perplexity=perplexity)

    # fit tsne
    trans_data = tsne.fit_transform(Xpca_for_tSNE).T

    # return tsne-transformed data
    return trans_data, colors_for_tSNE

#####################################################################
# Fit and apply UMAP
#####################################################################
def fit_and_apply_umap(profiles,n_neighbors=50,min_dist=0.0,frac=0.33):

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
