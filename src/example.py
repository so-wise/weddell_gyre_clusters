# import
import load_and_preprocess as lp
import bic_and_aic as ba
import plot_tools as pt
import file_io as io
import os.path
import density
import gmm
# import dask
from dask.distributed import Client
import dask

# start dask Client
#client = Client(n_workers=2, threads_per_worker=2, memory_limit='6GB')
#client

# set locations and names
descrip = 'allDomain' # extra description for filename
data_location = '../../so-chic-data/' # input data location
ploc = 'plots/'
dloc = 'models/'

# maximum number of components
max_N = 20

# calculate BIC and AIC?
getBIC = False

# save the processed output as a NetCDF file?
saveOutput = True

# longitude and latitude range
lon_min = -80
lon_max =  80
lat_min = -85
lat_max = -30

# depth range
zmin = 100.0
zmax = 900.0

# make decision about n_components_selected (iterative part of analysis)
n_components_selected = 10

# create filename for saving GMM and saving labelled profiles
gmm_fname = 'models/gmm_' + str(int(lon_min)) + 'to' + str(int(lon_max)) + 'lon_' + str(int(lat_min)) + 'to' + str(int(lat_max)) + 'lat_' + str(int(zmin)) + 'to' + str(int(zmax)) + 'depth_' + str(int(n_components_selected)) + 'K_' + descrip
fname = 'profiles_' + str(int(lon_min)) + 'to' + str(int(lon_max)) + 'lon_' + str(int(lat_min)) + 'to' + str(int(lat_max)) + 'lat_' + str(int(zmin)) + 'to' + str(int(zmax)) + 'depth_' + str(int(n_components_selected)) + 'K_' + descrip + '.nc'

# load profile subset based on ranges given above
profiles = lp.load_profile_data(data_location, lon_min, lon_max,
                                lat_min, lat_max, zmin, zmax)

# preprocess date and time
profiles = lp.preprocess_time_and_date(profiles)

# calculate conservative temperature, absolute salinity, and density (sig0)
profiles = density.calc_density(profiles)

# quick prof_T and prof_S selection plots
pt.prof_TS_sample_plots(ploc, profiles)

# apply PCA
profiles, pca, Xpca = lp.fit_and_apply_pca(profiles)

# calculate BIC and AIC
if getBIC==True:
    bic_mean, bic_std, aic_mean, aic_std = ba.calc_bic_and_aic(Xpca, max_N)
    pt.plot_bic_scores(ploc, max_N, bic_mean, bic_std)
    pt.plot_aic_scores(ploc, max_N, aic_mean, aic_std)

# if GMM exists, load it. Otherwise, create it.
if os.path.isfile(gmm_fname):
    best_gmm = io.load_gmm(gmm_fname)
else:
    best_gmm = gmm.train_gmm(Xpca, n_components_selected)
    io.save_gmm(gmm_fname, best_gmm)

# apply eitehr loaded or created GMM
profiles = gmm.apply_gmm(profiles, Xpca, best_gmm, n_components_selected)

# calculate class statistics
class_means, class_stds = gmm.calc_class_stats(profiles)

# plot T, S vertical structure of the classes,
pt.plot_CT_class_structure(ploc, profiles, class_means,
                           class_stds, n_components_selected, zmin, zmax)
pt.plot_SA_class_structure(ploc, profiles, class_means,
                           class_stds, n_components_selected, zmin, zmax)
pt.plot_sig0_class_structure(ploc, profiles, class_means,
                             class_stds, n_components_selected, zmin, zmax)

# plot label map
pt.plot_label_map(ploc, profiles, n_components_selected,
                   lon_min, lon_max, lat_min, lat_max)

# calculate the i-metric_
df1D = profiles.isel(depth=0)
gmm.calc_i_metric(profiles)
plt.plot_i_metric_single_panel(df1D)
plot_i_metric_multiple_panels(df1D, n_components_selected)
