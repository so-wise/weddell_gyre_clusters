
#####################################################################
# TO BE RUN ON THE WEDDELL CLUSTER: SUB-CLUSTERING
#####################################################################

# data file
#fname = 'models/profiles_-65to80lon_-85to-30lat_100to1000depth_12K_allDomain_density_test.nc'

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
import analysis
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
### import dask
#from dask.distributed import Client
#import dask

#####################################################################
# Start Dask client (not working at present)
#####################################################################
#client = Client(n_workers=2, threads_per_worker=2, memory_limit='6GB')
#client

#####################################################################
# Set runtime parameters (filenames, flags, ranges)
#####################################################################

# set locations and names
descrip = 'WeddellClass' # extra description for filename
data_location = '../../so-chic-data/' # input data location
classified_data_location =  'models/profiles_-65to80lon_-85to-30lat_100to1000depth_12K_allDomain_density_test.nc'
ploc = 'plots_WeddellClass_test/'
dloc = 'models/'

# if plot directory doesn't exist, create it
if not os.path.exists(ploc):
    os.makedirs(ploc)

# single class from previous effort to sub-classify
myClass=6

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
n_components_selected = 10

#longitude and latitude range
lon_min = -65
lon_max =  80
lat_min = -85
lat_max = -30
# depth range
zmin = 100.0
zmax = 1000.0
# density range
sig0range = (26.0, 27.0)

# temperature and salinity ranges for plotting
Trange=(-2, 2.0)
Srange=(34.0, 35.0)
sig0min=27.0
sig0max=28.0
# based on the above, calculate the density range
#sig0min = round(density.calc_scalar_density(Trange[0],Srange[0],
#    p=0.0,lon=0.0,lat=-60),2)
#sig0max = round(density.calc_scalar_density(Trange[1],Srange[1],
#    p=0.0,lon=0.0,lat=-60),2)

# create filename for saving GMM and saving labelled profiles
pca_fname = dloc + 'pca_' + str(int(lon_min)) + 'to' + str(int(lon_max)) + 'lon_' + str(int(lat_min)) + 'to' + str(int(lat_max)) + 'lat_' + str(int(zmin)) + 'to' + str(int(zmax)) + 'depth_' + str(int(n_pca)) + descrip
gmm_fname = dloc + 'gmm_' + str(int(lon_min)) + 'to' + str(int(lon_max)) + 'lon_' + str(int(lat_min)) + 'to' + str(int(lat_max)) + 'lat_' + str(int(zmin)) + 'to' + str(int(zmax)) + 'depth_' + str(int(n_components_selected)) + 'K_' + descrip
fname = dloc + 'profiles_' + str(int(lon_min)) + 'to' + str(int(lon_max)) + 'lon_' + str(int(lat_min)) + 'to' + str(int(lat_max)) + 'lat_' + str(int(zmin)) + 'to' + str(int(zmax)) + 'depth_' + str(int(n_components_selected)) + 'K_' + descrip + '.nc'

# colormap
colormap = plt.get_cmap('tab20', n_components_selected)

#####################################################################
# Run the standard analysis stuff
#####################################################################
#####################################################################
# Data loading and preprocessing
#####################################################################

# load profile subset based on ranges given above
profiles = lp.load_single_class(classified_data_location, selected_class=myClass)

# get rid of old high-z and sigma interpolation, redo it
profiles = profiles.drop({'sa_on_highz','ct_on_highz','ct_on_sig0','sa_on_sig0'})
profiles = profiles.drop_dims({'depth_highz','sig0_levs'})
profiles = lp.regrid_onto_more_vertical_levels(profiles, zmin, zmax)
profiles = lp.regrid_onto_density_levels(profiles)

#####################################################################
# Dimensionality reduction / transformation
#####################################################################

# use PCA, either regular or KernelPCA
if transform_method=='pca':

    # if trained PCA already exists, load it
    if os.path.isfile(pca_fname):
        pca = io.load_pca(pca_fname)
        Xtrans = lp.apply_pca(profiles, pca, method='onSig')
    # otherwise, go ahead and train it
    else:
        # apply PCA
        pca, Xtrans = lp.fit_and_apply_pca(profiles,
                                           number_of_pca_components=n_pca,
                                           kernel=use_kernel_pca,
                                           train_frac=0.99)
        # save for future use
        io.save_pca(pca_fname, pca)

    # plot PCA structure
    #pt.plot_pca_vertical_structure(ploc, profiles, pca, Xtrans)
    pt.plot_pca3D(ploc, colormap, profiles, Xtrans, frac=0.33)

    # pairplot of transformed variables
    pt.plot_pairs(ploc, Xtrans, kind='hist', descr=transform_method)

# the UMAP method produces a 2D projection
elif transform_method=='umap':

    # alternatively, apply UMAP
    embedding, Xtrans = lp.fit_and_apply_umap(profiles,
                                              n_neighbors=50, min_dist=0.0)

    # plot UMAP structure
    pt.plot_umap(ploc, Xtrans)

    # pairplot of transformed variables
    pt.plot_pairs(ploc, Xtrans, kind='hist', descr=transform_method)

else:

    print('Invalid transform method! Must be pca or umap')

#####################################################################
# Statistical measures to inform number of classes
#####################################################################

# calculate BIC and AIC
if getBIC==True:
    bic_mean, bic_std, aic_mean, aic_std = ba.calc_bic_and_aic(Xtrans, max_N)
    pt.plot_bic_scores(ploc, max_N, bic_mean, bic_std)
    pt.plot_aic_scores(ploc, max_N, aic_mean, aic_std)

#####################################################################
# Establish GMM (either load it or train a new one)
#####################################################################

# if GMM exists, load it. Otherwise, create it.
if os.path.isfile(gmm_fname):
    best_gmm = io.load_gmm(gmm_fname)
else:
    best_gmm = gmm.train_gmm(Xtrans, n_components_selected)
    io.save_gmm(gmm_fname, best_gmm)

# apply either loaded or created GMM
profiles = gmm.apply_gmm(profiles, Xtrans, best_gmm, n_components_selected)

# calculate class statistics
class_means, class_stds = gmm.calc_class_stats(profiles)

#####################################################################
# Plot classification results
#####################################################################

# plot label map
pt.plot_label_map(ploc, profiles, n_components_selected,
                   lon_min, lon_max, lat_min, lat_max)
