#####################################################################
# These may need to be installed
#####################################################################
# pip install umap-learn
# pip install seaborn
# pip install gsw

#####################################################################
# Import packages
#####################################################################

### modules in this package
import load_and_preprocess as lp
import bic_and_aic as ba
import plot_tools as pt
import file_io as io
import xarray
import density
import gmm
### plotting tools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
### os tools
import os.path

# import pakcages
import numpy as np
import xarray as xr
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn import manifold
from xgcm import Grid

import file_io as io

#####################################################################
# Set runtime parameters (filenames, flags, ranges)
#####################################################################

# set locations and names
descrip = 'try_scaling' # extra description for filename
data_location = '../../so-chic-data/' # input data location
ploc = 'plots_try_scaling/'
dloc = 'models/'

# if plot directory doesn't exist, create it
if not os.path.exists(ploc):
    os.makedirs(ploc)

# calculate BIC and AIC? set max number of components
getBIC = False
max_N = 20

# transformation method (pca, umap)
# --- at present, UMAP transform crashes the kernel
transform_method = 'pca'

# use the kernel PCA approach (memory intensive, not working yet)
use_kernel_pca = False

# save the processed output as a NetCDF file?
saveOutput = True

# number of PCA components
n_pca = 6

# make decision about n_components_selected (iterative part of analysis)
n_components_selected = 8

# longitude and latitude range
lon_min = -80
lon_max =  80
lat_min = -85
lat_max = -30
# depth range
zmin = 100.0
zmax = 900.0

# temperature and salinity ranges for plotting
Trange=(-3.0, 20.0)
Srange=(33.0, 35.0)
# based on the above, calculate the density range
sig0min = round(density.calc_scalar_density(Trange[0],Srange[0],
    p=0.0,lon=0.0,lat=-60),2)
sig0max = round(density.calc_scalar_density(Trange[1],Srange[1],
    p=0.0,lon=0.0,lat=-60),2)

# create filename for saving GMM and saving labelled profiles
pca_fname = dloc + 'pca_' + str(int(lon_min)) + 'to' + str(int(lon_max)) + 'lon_' + str(int(lat_min)) + 'to' + str(int(lat_max)) + 'lat_' + str(int(zmin)) + 'to' + str(int(zmax)) + 'depth_' + str(int(n_pca)) + descrip
gmm_fname = dloc + 'gmm_' + str(int(lon_min)) + 'to' + str(int(lon_max)) + 'lon_' + str(int(lat_min)) + 'to' + str(int(lat_max)) + 'lat_' + str(int(zmin)) + 'to' + str(int(zmax)) + 'depth_' + str(int(n_components_selected)) + 'K_' + descrip
fname = dloc + 'profiles_' + str(int(lon_min)) + 'to' + str(int(lon_max)) + 'lon_' + str(int(lat_min)) + 'to' + str(int(lat_max)) + 'lat_' + str(int(zmin)) + 'to' + str(int(zmax)) + 'depth_' + str(int(n_components_selected)) + 'K_' + descrip + '.nc'

# colormap
colormap = plt.get_cmap('tab10', n_components_selected)

#####################################################################
# Data loading and preprocessing
#####################################################################

# load the ctds, floats, and seals
ctds = xr.open_mfdataset(data_location + 'CTD/*.nc',
                         concat_dim='iPROF', combine='nested')
floats = xr.open_mfdataset(data_location + 'FLOATS/*.nc',
                           concat_dim='iPROF', combine='nested')
seals = xr.open_mfdataset(data_location + 'SEALS/*.nc',
                          concat_dim='iPROF', combine='nested')

# add variable to indicate data source
ctds['source'] = 'ctd'
floats['source'] = 'float'
seals['source'] = 'seal'

# combine into single xarray.Dataset object
profiles = xr.combine_nested([ctds, floats, seals],
                             concat_dim='iPROF')

# assign depth coordinate
profiles.coords['iDEPTH'] = profiles.prof_depth[0,:].values

# select subset of data between 0-1000 dbar
#profiles = profiles.sel(iDEPTH=slice(zmin,zmax))

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

#
# TEST OUT THE SCALING IDEA ON A SINGLE PROFILE
#

# Profiles don't necessarily extend all the way down to the
# bathymetric depth.

# select a single profile for testing
#p1 = profiles.isel(profile=slice(1000,1010))
p1 = profiles.isel(profile=1000)

ds = io.load_bathymetry()
bath_lon=ds['lon'][:]
bath_lat=ds['lat'][:]
bathy=ds['bathy'][:]
# set maximum depth scale
#bathy[bathy<-1000]=-1000

# use bathymetry for scaling
da=xr.DataArray(data=bathy,
        dims=("lat","lon"),
        coords={"lat":bath_lat,"lon":bath_lon})

# interpolation works! For example:
# da.interp(lat=-60,lon=0)

bathy_at_p1 = abs(da.interp(lat=p1.lat.values,
                  lon=p1.lon.values)).values

depth_scaled = p1.depth.values/bathy_at_p1

p1 = p1.assign_coords({"depth_scaled": ("depth", depth_scaled)})

# now interpolate onto a standard set of levels

# define grid object
grid = Grid(p1, coords={'Z': {'center': 'depth'}}, periodic=False)
target_z_levels = np.linspace(0.0, 1.0, 100)

# linearly interpolate temperature onto selected z levels
prof_T_trans = grid.transform(p1.prof_T, 'Z',
                              target_z_levels,
                              target_data=p1.depth_scaled,
                              method='linear',
                              mask_edges=False)

# the above sort of works, except with one big flaw:
# some profiles don't extend to bathymetry.

# We need to find the first non-NaN value and scale by that.
# Except that's a bit random: many different factors affect how
# deep a particular profile might be. Given that, this scaling will
# quickly turn into nonsense.

# Why scale by the "total length of profile" when that's not a physicaslly
# meaningful quantity? Why scale by "how deep the seal or CTD happened to go that time"
# It's not physical or informative.

# I'll keep this file for future reference, but for now, this idea is gone.
