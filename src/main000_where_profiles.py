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
ploc = 'plots/plots_allDomain_top1000m_K05_forOSM22/'

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

# isolate profiles from different sources
argo = profiles.where(profiles.source=='argo',drop=True)
ctd = profiles.where(profiles.source=='ctd',drop=True)
seal = profiles.where(profiles.source=='seal',drop=True)

# plot histogram of profiles locations
pt.plot_histogram_of_profile_locations(ploc, profiles, lon_range, lat_range, source='argo', binsize=2)
pt.plot_histogram_of_profile_locations(ploc, profiles, lon_range, lat_range, source='ctd', binsize=2)
#pt.plot_histogram_of_profile_locations(ploc, profiles, lon_range, lat_range, source='seal')
pt.plot_histogram_of_profile_locations(ploc, profiles, lon_range, lat_range, source='all', binsize=2)
