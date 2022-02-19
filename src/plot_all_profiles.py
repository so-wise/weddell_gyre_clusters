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

#####################################################################
# Set runtime parameters (filenames, flags, ranges)
#####################################################################

# set locations and names
descrip = 'allDomain' # extra description for filename
data_location = '../../so-chic-data/' # input data location
ploc = 'plots/plots_forAGU/'
dloc = 'models/'

# if plot directory doesn't exist, create it
if not os.path.exists(ploc):
    os.makedirs(ploc)

#longitude and latitude range
lon_min = -65
lon_max =  80
lat_min = -85
lat_max = -30
# depth range
zmin = 300.0
zmax = 1000.0
# ranges
sig0range = (26.0, 28.0)
Trange=(-2, 18.0)
Srange=(34.0, 36.0)

#####################################################################
# Data loading and preprocessing
#####################################################################

# load profile subset based on ranges given above
profiles = lp.load_profile_data(data_location, lon_min, lon_max,
                                lat_min, lat_max, zmin, zmax)

# preprocess date and time
profiles = lp.preprocess_time_and_date(profiles)

# calculate conservative temperature, absolute salinity, and density (sig0)
profiles = density.calc_density(profiles)

# quick prof_T and prof_S selection plots
pt.prof_TS_sample_plots(ploc, profiles)

# plot random profile
pt.plot_profile(ploc, profiles.isel(profile=2000))

# regrid onto density levels (maybe useful for plotting later?)
profiles = lp.regrid_onto_more_vertical_levels(profiles, zmin, zmax)
profiles = lp.regrid_onto_density_levels(profiles)

# plot many profiles
pt.plot_many_profiles(ploc, df, frac=0.01, ymin=20, ymax=1000,
                       sig0min=23.0 sig0max=28.0

#####################################################################
# END
#####################################################################
