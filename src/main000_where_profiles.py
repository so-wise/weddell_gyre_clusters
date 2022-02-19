#####################################################################
# Plot profile locations, get statistics of profiles, sources, etc.
#####################################################################

#####################################################################
# These may need to be installed
#####################################################################
# pip install umap-learn seaborn gsw cmocean

#####################################################################
# Import packages
#####################################################################

### modules in this package
import load_and_preprocess as lp
import plot_tools as pt
import file_io as io
import density
import xarray
### plotting tools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
### os tools
import os.path

#####################################################################
# Set runtime parameters (filenames, flags, ranges)
#####################################################################

# set locations and names
descrip = 'allDomain' # extra description for filename
data_location = '../../so-chic-data/' # input data location
ploc = 'plots/plots_allDomain_top1000m_K05_forOSM22_dev/'

# if plot directory doesn't exist, create it
if not os.path.exists(ploc):
    os.makedirs(ploc)

#longitude and latitude range
lon_min = -65
lon_max =  80
lat_min = -85
lat_max = -30
# depth range
zmin = 20.0
zmax = 1000.0

# ranges
lon_range = (lon_min, lon_max)
lat_range = (lat_min, lat_max)
depth_range = (zmin, zmax)

# temperature, salinity, and density ranges for plotting
Trange = (-2, 25.0)
Srange = (33.0, 37.0)
sig0range = (23.0, 28.0)

#####################################################################
# Data loading and preprocessing
#####################################################################

# load profile subset based on ranges given above
profiles = lp.load_profile_data(data_location, lon_min, lon_max,
                                lat_min, lat_max, zmin, zmax)

# preprocess date and time
profiles = lp.preprocess_time_and_date(profiles)
profiles = density.calc_density(profiles)
profiles = lp.regrid_onto_more_vertical_levels(profiles, zmin, zmax)
profiles = lp.regrid_onto_density_levels(profiles)

# print some values : how many profiles?
n_argo = profiles.where(profiles.source=='argo',drop=True).profile.size
n_ctd = profiles.where(profiles.source=='ctd',drop=True).profile.size
n_seal = profiles.where(profiles.source=='seal',drop=True).profile.size
n_profiles = n_argo + n_ctd + n_seal
print('******************************************************************')
print('Number of Argo profiles after selection applied = ' + str(n_argo))
print('Number of CTD profiles after selection applied = ' + str(n_ctd))
print('Number of seal profiles after selection applied = ' + str(n_seal))
print('******************************************************************')
print('Total number of profiles after selection applied = ' + str(n_profiles))

# make the "all profiles" plot
pt.plot_many_profiles(ploc, profiles, frac = 0.05,
                      zmin = zmin, zmax = zmax,
                      Tmin = Trange[0], Tmax = Trange[1],
                      Smin = Srange[0], Smax = Srange[1],
                      sig0min = sig0range[0], sig0max = sig0range[1],
                      alpha = 0.01, modStr = '',
                      colorVal = 'black')

# isolate profiles from different sources
argo = profiles.where(profiles.source=='argo',drop=True)
ctd = profiles.where(profiles.source=='ctd',drop=True)
seal = profiles.where(profiles.source=='seal',drop=True)

# plot histogram of profiles locations
pt.plot_histogram_of_profile_locations(ploc, profiles, lon_range, lat_range, source='argo', binsize=2)
pt.plot_histogram_of_profile_locations(ploc, profiles, lon_range, lat_range, source='ctd', binsize=2)
#pt.plot_histogram_of_profile_locations(ploc, profiles, lon_range, lat_range, source='seal')
pt.plot_histogram_of_profile_locations(ploc, profiles, lon_range, lat_range, source='all', binsize=2)

# make a separate colorbar


#####################################################################
# Data loading and plotting (Antarctic Class Only)
#####################################################################

# set locations and names
descrip = 'WeddellOnly' # extra description for filename
data_location = '../../so-chic-data/' # input data location
classified_data_location = 'models/profiles_-65to80lon_-85to-30lat_20to1000depth_5K_allDomain_revised.nc'
ploc = 'plots/plots_WeddellClassOnly_top1000m_K04_forOSM22_dev/'
dloc = 'models/'

# single class from previous effort to sub-classify
# don't forget 0 indexing
myClass=1

#longitude and latitude range
lon_min = -65
lon_max =  80
lat_min = -80
lat_max = -45
# depth range
zmin = 20.0
zmax = 1000.0

# ranges
lon_range = (lon_min, lon_max)
lat_range = (lat_min, lat_max)
depth_range = (zmin, zmax)

# temperature and salinity ranges for plotting
Trange=(-2.2, 6.0)
Srange=(33.5, 35.0)
sig0range = (26.6, 28.0)

# load single class (just the Weddell One)
profiles_antarctic = lp.load_single_class(classified_data_location, selected_class=myClass)

# histogram of profile locations
pt.plot_histogram_of_profile_locations(ploc, profiles_antarctic, lon_range, lat_range, source='all', binsize=2)

# -- the saved dataset doesn't have the "source" variabile, unfortunately (it's mostly Argo, anyway)
#pt.plot_histogram_of_profile_locations(ploc, profiles_antarctic, lon_range, lat_range, source='ctd', binsize=2)
#pt.plot_histogram_of_profile_locations(ploc, profiles_antarctic, lon_range, lat_range, source='argo', binsize=2)

# make the "all profiles" plot
pt.plot_many_profiles(ploc, profiles_antarctic, frac = 0.10,
                      zmin = zmin, zmax = zmax,
                      Tmin = Trange[0], Tmax = Trange[1],
                      Smin = Srange[0], Smax = Srange[1],
                      sig0min = sig0range[0], sig0max = sig0range[1],
                      alpha = 0.01, modStr = '',
                      colorVal = 'black')
