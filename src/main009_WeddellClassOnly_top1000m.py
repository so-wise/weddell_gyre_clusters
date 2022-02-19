#####################################################################
# Import packages
#####################################################################

### modules in this package
import load_and_preprocess as lp
import analysis as at
import bic_and_aic as ba
import plot_tools as pt
import file_io as io
import numpy as np
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

# Suppress a particular warning
import warnings
warnings.filterwarnings('ignore', 'RuntimeWarning: All-NaN slice encountered')

#####################################################################
# Set runtime parameters (filenames, flags, ranges)
#####################################################################

# set locations and names
descrip = 'WeddellOnly' # extra description for filename
data_location = '../../so-chic-data/' # input data location
classified_data_location = 'models/profiles_-65to80lon_-85to-30lat_20to1000depth_5K_allDomain_revised.nc'
ploc = 'plots/plots_WeddellClassOnly_top1000m_K04_forOSM22_dev/'
dloc = 'models/'

# if plot directory doesn't exist, create it
if not os.path.exists(ploc):
    os.makedirs(ploc)

# single class from previous effort to sub-classify
# don't forget 0 indexing
myClass=1

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
n_components_selected = 4

#longitude and latitude range
lon_min = -65
lon_max =  80
lat_min = -80
lat_max = -45
# depth range
zmin = 20.0
zmax = 1000.0
# density range
sig0range = (26.6, 28.0)

# temperature and salinity ranges for plotting
lon_range=(lon_min, lon_max)
lat_range=(lat_min, lat_max)
Trange=(-2.2, 6.0)
Srange=(33.5, 35.0)

# create filename for saving GMM and saving labelled profiles
pca_fname = dloc + 'pca_' + str(int(lon_min)) + 'to' + str(int(lon_max)) + 'lon_' + str(int(lat_min)) + 'to' + str(int(lat_max)) + 'lat_' + str(int(zmin)) + 'to' + str(int(zmax)) + 'depth_' + str(int(n_pca)) + descrip
gmm_fname = dloc + 'gmm_' + str(int(lon_min)) + 'to' + str(int(lon_max)) + 'lon_' + str(int(lat_min)) + 'to' + str(int(lat_max)) + 'lat_' + str(int(zmin)) + 'to' + str(int(zmax)) + 'depth_' + str(int(n_components_selected)) + 'K_' + descrip
fname = dloc + 'profiles_' + str(int(lon_min)) + 'to' + str(int(lon_max)) + 'lon_' + str(int(lat_min)) + 'to' + str(int(lat_max)) + 'lat_' + str(int(zmin)) + 'to' + str(int(zmax)) + 'depth_' + str(int(n_components_selected)) + 'K_' + descrip + '.nc'

# colormap
colormap = plt.get_cmap('tab10', n_components_selected)
colormap_cividis = plt.get_cmap('cividis', 20)

#####################################################################
# Run the standard analysis stuff
#####################################################################

#####################################################################
# Data loading and preprocessing
#####################################################################

# load single class (just the Weddell One)
profiles = lp.load_single_class(classified_data_location, selected_class=myClass)

# quick prof_T and prof_S selection plots
pt.prof_TS_sample_plots(ploc, profiles)

# plot random profile
pt.plot_profile(ploc, profiles.isel(profile=1000))

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

# simplify Dataset for plotting purposes
dfp = profiles
dfp = dfp.drop({'depth_highz','sig0_levs','prof_T','prof_S','ct_on_highz',
                'sa_on_highz','sig0_on_highz','ct_on_sig0','sa_on_sig0'})

# plot T, S vertical structure of the classes
pt.plot_class_vertical_structures(ploc, profiles, n_components_selected, colormap,
                                  zmin=zmin, zmax=zmax,
                                  Tmin=Trange[0], Tmax=Trange[1],
                                  Smin=Srange[0], Smax=Srange[1],
                                  sig0min=sig0range[0], sig0max=sig0range[1],
                                  frac=0.33)

# TS diagram just showing the mean values
pt.plot_TS_withMeans(ploc, class_means, class_stds, n_components_selected, colormap,
                     PTrange=Trange, SPrange=Srange)

# CT, SA, and sig0 class structure (means and standard deviation)
pt.plot_CT_class_structure(ploc, dfp, class_means, class_stds,
                           n_components_selected, colormap, zmin, zmax,
                           Tmin=Trange[0], Tmax=Trange[1])
pt.plot_SA_class_structure(ploc, dfp, class_means, class_stds,
                           n_components_selected, colormap, zmin, zmax,
                           Smin=Srange[0], Smax=Srange[1])
pt.plot_sig0_class_structure(ploc, dfp, class_means, class_stds,
                           n_components_selected, colormap, zmin, zmax,
                           sig0min=sig0range[0], sig0max=sig0range[1])
pt.plot_CT_and_SA_class_structure(ploc, profiles, class_means, class_stds,
                                  n_components_selected, colormap, zmin, zmax,
                                  Tmin=Trange[0], Tmax=Trange[1],
                                  Smin=Srange[0], Smax=Srange[1])

# plot 3D pca structure (now with class labels)
pt.plot_pca3D(ploc, colormap, dfp, Xtrans, frac=0.33, withLabels=True)

# plot some single level T-S diagrams
pt.plot_TS_single_lev(ploc, dfp, n_components_selected, colormap,
                      descrip='', plev=0, PTrange=Trange,
                      SPrange=Srange, lon = -20, lat = -65, rr = 0.60)

# plot multiple-level T-S diagrams (one for each class)
pt.plot_TS_multi_lev(ploc, dfp, n_components_selected, colormap=colormap_cividis,
                     descrip='', plev=0, PTrange=Trange,
                     SPrange=Srange, lon = -20, lat = -65, rr = 0.33)

# plot T-S diagram (all levels shown)
pt.plot_TS_all_lev(ploc, dfp, n_components_selected, colormap,
                   descrip='', PTrange=Trange, SPrange=Srange,
                   lon = -20, lat = -65, rr = 0.33)

# plot T-S diagrams (by class, shaded by year and month)
pt.plot_TS_bytime(ploc, dfp, n_components_selected,
                   descrip='', PTrange=Trange, SPrange=Srange,
                   lon = -20, lat = -65, rr = 0.60, timeShading='year')
pt.plot_TS_bytime(ploc, dfp, n_components_selected,
                   descrip='', PTrange=Trange, SPrange=Srange,
                   lon = -20, lat = -65, rr = 0.60, timeShading='month')

# plot label map
pt.plot_label_map(ploc, dfp, n_components_selected, colormap,
                   lon_min, lon_max, lat_min, lat_max)

# Calc Tmin, Tmax, Smin, Smax
dfp['Tmin'] = dfp.prof_CT.min(dim='depth')
dfp['Tmax'] = dfp.prof_CT.max(dim='depth')
dfp['Smin'] = dfp.prof_SA.min(dim='depth')
dfp['Smax'] = dfp.prof_SA.max(dim='depth')
dfp['sig0min'] = dfp.sig0.min(dim='depth')
dfp['sig0max'] = dfp.sig0.max(dim='depth')

# select the top pressure level for plotting purposes
df1D = dfp.isel(depth=0)

# calculate the i-metric
df1D = gmm.calc_i_metric(profiles)

# plot i-metric
pt.plot_i_metric_single_panel(ploc, df1D, lon_min, lon_max, lat_min, lat_max)
pt.plot_i_metric_multiple_panels(ploc, df1D, lon_min, lon_max,
                                 lat_min, lat_max, n_components_selected)
pt.plot_i_metric_multiple_panels(ploc, df1D, lon_min, lon_max,
                                 lat_min, lat_max, n_components_selected)

# i-metric, multiple panels, histogram style
pt.plot_i_metric_multiple_panels_hist(ploc, df1D, lon_min, lon_max,
                                 lat_min, lat_max, n_components_selected)

# surface temperatures and surface salinities, histogram style
# --- could probably replace with a single function that can be called by
# --- a text keyword for T, S, min, max, etc. (simple if conditoinal)
#pt.plot_hist_map_Tsurf(ploc, df1D, lon_min, lon_max,
#                       lat_min, lat_max, n_components_selected)
#pt.plot_hist_map_Tmax(ploc, df1D, lon_min, lon_max,
#                      lat_min, lat_max, n_components_selected)
#pt.plot_hist_map_Ssurf(ploc, df1D, lon_min, lon_max,
#                       lat_min, lat_max, n_components_selected)

# histogram map
pt.plot_hist_map(ploc, df1D, lon_range, lat_range,
                 n_components_selected,
                 c_range=(-2,2),
                 vartype='Tsurf',
                 colormap=plt.get_cmap('coolwarm'))

pt.plot_hist_map(ploc, df1D, lon_range, lat_range,
                 n_components_selected,
                 c_range=(0,3),
                 vartype='Tmax',
                 colormap=plt.get_cmap('coolwarm'))

# some T-S histograms
sbins = np.arange(Srange[0], Srange[1], 0.025)
tbins = np.arange(Trange[0], Trange[1], 0.1)
df_select = dfp.where(dfp.label==0, drop=True)
histTS_class1 = pt.calc_and_plot_volume_histogram_TS(ploc, df_select, sbins=sbins, tbins=tbins, modStr='class1')
df_select = dfp.where(dfp.label==1, drop=True)
histTS_class2 = pt.calc_and_plot_volume_histogram_TS(ploc, df_select, sbins=sbins, tbins=tbins, modStr='class2')
df_select = dfp.where(dfp.label==2, drop=True)
histTS_class3 = pt.calc_and_plot_volume_histogram_TS(ploc, df_select, sbins=sbins, tbins=tbins, modStr='class3')
df_select = dfp.where(dfp.label==3, drop=True)
histTS_class4 = pt.calc_and_plot_volume_histogram_TS(ploc, df_select, sbins=sbins, tbins=tbins, modStr='class4')

#####################################################################
# Further analysis of specific classes, regions, time variation
#####################################################################

# THIS STUFF NOT WORKING AT PRESENT

# Visualize profile stats by class and year (all profiles)
#at.examine_prof_stats_by_label_and_year(ploc, profiles, str, frac = 0.95, \
#                                        zmin=20, zmax=1000, \
#                                        Tmin = Trange[0], Tmax = Trange[1], \
#                                        Smin = Srange[0], Smax = Srange[1], \
#                                        sig0min = sig0range[0], sig0max = sig0range[1], \
#                                        alpha=0.1)

# Weddell-Scotia confluence waters
#box_edges=[-64.5, 40, -67, -50]
#df_wsc, df_not_wsc = at.split_single_class_by_box(profiles, class_split=3,
#                                                  box_edges=box_edges)

# Plot all the profiles in the box
#plocA = 'plots/plots_WeddellClassOnly_top1000m_K04_wsc_analysis_dev/'
#pt.plot_many_profiles(plocA, df_wsc, frac=0.95, zmin=20, zmax=1000,
#                      sig0min=27.0, sig0max=28.0, alpha=0.1)

# Visualize profile stats by class and year (all profiles)
#at.examine_prof_stats_by_label_and_year(plocA, df_wsc, str, frac = 0.95, \
#                                        zmin=20, zmax=1000, \
#                                        Tmin = -1.9, Tmax = 7.0, \
#                                        Smin = 33.5, Smax = 35.0, \
#                                        sig0min = 26.8, sig0max = 28.0, \
#                                        alpha=0.1)

#####################################################################
# Save the profiles in a separate NetCDF file
#####################################################################

if saveOutput==True:
    profiles.to_netcdf(fname, mode='w')

#####################################################################
# END
#####################################################################
