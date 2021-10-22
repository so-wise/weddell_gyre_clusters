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

#####################################################################
# Set runtime parameters (filenames, flags, ranges)
#####################################################################

# set locations and names
descrip = 'WeddellSeaZoom01' # extra description for filename
data_location = '../../so-chic-data/' # input data location
ploc = 'plots/plots_WeddellSeaZoom01_10m-300m_K10/'
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
saveOutput = False

# number of PCA components
n_pca = 6

# make decision about n_components_selected (iterative part of analysis)
n_components_selected = 10

#longitude and latitude range
lon_min = -70
lon_max =  -30
lat_min = -85
lat_max = -60
# depth range
zmin = 10.0
zmax = 300.0
# density range
sig0range = (26.75, 28.0)

# temperature and salinity ranges for plotting
Trange=(-2.2, 3.0)
Srange=(33.5, 35.0)
# based on the above, calculate the density range
# sig0a = round(density.calc_scalar_density(Trange[0],Srange[0],
#     p=0.0,lon=0.0,lat=-60),2)
# sig0b = round(density.calc_scalar_density(Trange[1],Srange[1],
#     p=0.0,lon=0.0,lat=-60),2)

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

# pairplot: unscaled (warning: this is very slow)
#pt.plot_pairs(ploc,np.concatenate((profiles.prof_CT, profiles.prof_SA),axis=1),
#              kind="hist",descr="unscaled")

# select more specific density range
#profiles = lp.select_sig0_range(profiles, sig0range=sig0range)

# plot sa and ct on density levels (CRASHES)
#pt.plot_profiles_on_density_levels(ploc, profiles)

#####################################################################
# Dimensionality reduction / transformation
#####################################################################

# use PCA, either regular or KernelPCA
if transform_method=='pca':

    # if trained PCA already exists, load it
    if os.path.isfile(pca_fname):
        pca = io.load_pca(pca_fname)
        Xtrans = lp.apply_pca(profiles, pca)
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
# Calculate and plot tSNE with class labels
#####################################################################

# fit and apply tsne
tSNE_data, colors_for_tSNE = lp.fit_and_apply_tsne(profiles, Xtrans)

# plot t-SNE with class labels
pt.plot_tsne(ploc, colormap, tSNE_data, colors_for_tSNE)

#####################################################################
# Plot classification results
#####################################################################

# plot T, S vertical structure of the classes
pt.plot_CT_class_structure(ploc, profiles, class_means,class_stds,
                           n_components_selected, zmin, zmax,
                           Tmin=Trange[0], Tmax=Trange[1])
pt.plot_SA_class_structure(ploc, profiles, class_means,class_stds,
                           n_components_selected, zmin, zmax,
                           Smin=Srange[0], Smax=Srange[1])
pt.plot_sig0_class_structure(ploc, profiles, class_means,class_stds,
                           n_components_selected, zmin, zmax)

# plot T, S vertical structure on sig0 surfaces
# pt.plot_CT_class_structure_onSig(ploc, profiles, class_means,class_stds,
#                           n_components_selected,
#                           Tmin=Trange[0], Tmax=Trange[1])
# pt.plot_SA_class_structure_onSig(ploc, profiles, class_means,class_stds,
#                           n_components_selected,
#                           Smin=Srange[0], Smax=Srange[1])

# plot 3D pca structure (now with class labels)
pt.plot_pca3D(ploc, colormap, profiles, Xtrans, frac=0.33, withLabels=True)

# plot some single level T-S diagrams
pt.plot_TS_single_lev(ploc, profiles, n_components_selected,
                      descrip='', plev=0, PTrange=Trange,
                      SPrange=Srange, lon = -20, lat = -65, rr = 0.60)

# plot multiple-level T-S diagrams
pt.plot_TS_multi_lev(ploc, profiles, n_components_selected,
                     descrip='', plev=0, PTrange=Trange,
                     SPrange=Srange, lon = -20, lat = -65, rr = 0.60)

# plot T-S diagram (all levels shown)
pt.plot_TS_all_lev(ploc, profiles, n_components_selected,
                   descrip='', PTrange=Trange, SPrange=Srange,
                   lon = -20, lat = -65, rr = 0.60)

# plot T-S diagrams (by class, shaded by year and month)
pt.plot_TS_bytime(ploc, profiles, n_components_selected,
                   descrip='', PTrange=Trange, SPrange=Srange,
                   lon = -20, lat = -65, rr = 0.60, timeShading='year')
pt.plot_TS_bytime(ploc, profiles, n_components_selected,
                   descrip='', PTrange=Trange, SPrange=Srange,
                   lon = -20, lat = -65, rr = 0.60, timeShading='month')

# plot label map
pt.plot_label_map(ploc, profiles, n_components_selected,
                   lon_min, lon_max, lat_min, lat_max)

# calculate the i-metric
df1D = profiles.isel(depth=0)
df1D = gmm.calc_i_metric(profiles)
pt.plot_i_metric_single_panel(ploc, df1D, lon_min, lon_max, lat_min, lat_max)
pt.plot_i_metric_multiple_panels(ploc, df1D, lon_min, lon_max,
                                 lat_min, lat_max, n_components_selected)

#####################################################################
# Save the profiles in a separate NetCDF file
#####################################################################

if saveOutput==True:
    profiles.to_netcdf(fname, mode='w')

#####################################################################
# END
#####################################################################
