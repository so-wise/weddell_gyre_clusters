#####################################################################
# Utilities for loading profile data, slicing
#####################################################################

import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
import matplotlib.patches as mpatches
import pandas as pd
from sklearn import manifold
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import cmocean
from xhistogram.xarray import histogram
### os tools
import os.path
from glob import glob
import file_io as io
import density
import random
import gsw

#####################################################################
# Plot histogram of profile locations
#####################################################################
def plot_histogram_of_profile_locations(ploc, profiles, lon_range, lat_range,
                                        source='all', binsize=2,
                                        bathy_fname="bathy.nc",
                                        lev_range=range(-6000,1,500),
                                        myPlotLevels=30, vmin=0, vmax=200):
#
# source : may be 'argo', 'ctd', 'seal', or 'all'
# binsize : size of  lat-lon bins in degrees
#

    # print
    print("plot_tools.plot_histogram_of_profile_locations")

    # select
    if source=='all':
        df = profiles
    else:
        df = profiles.where(profiles.source==source, drop=True)

    # bins
    lon_bins = np.arange(lon_range[0], lon_range[1], binsize)
    lat_bins = np.arange(lat_range[0], lat_range[1], binsize)

    # histogram
    hLatLon = histogram(df.lon, df.lat, bins=[lon_bins, lat_bins])

    # load bathymetry
    bds = io.load_bathymetry(bathy_fname)
    bathy_lon = bds['lon'][:]
    bathy_lat = bds['lat'][:]
    bathy = bds['bathy'][:]

    #
    #-- original attempt
    #

    # cartopy plot
    plt.figure(figsize=(17, 13))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                    ccrs.PlateCarree())
    # colormesh histogram
    CS = plt.pcolormesh(lon_bins, lat_bins, hLatLon.T, transform=ccrs.PlateCarree())
    plt.clim(vmin, vmax)
    ax.coastlines(resolution='50m',color='white')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    ax.add_feature(cartopy.feature.LAND)
    plt.savefig(ploc + 'histogram_lat_lon_map_' + source + '.png', bbox_inches='tight')
    plt.savefig(ploc + 'histogram_lat_lon_map_' + source + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # separate colorbar
    a = np.array([[vmin,vmax]])
    plt.figure(figsize=(9, 1.5))
    img = plt.imshow(a, cmap="viridis")
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])
    cbar = plt.colorbar(orientation="horizontal", cax=cax)
    cbar.ax.tick_params(labelsize=22)
    plt.savefig(ploc + 'histogram_latlon_map_colorbar.png', bbox_inches='tight')
    plt.savefig(ploc + 'histogram_latlon_map_colorbar.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot single profile
#####################################################################
def plot_profile(ploc, df):
   print("plot_tools.plot_profiles")
   fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=[24,5])
   df.prof_T.plot(ax=ax1, y='depth', yincrease=False)
   df.prof_S.plot(ax=ax2, y='depth', yincrease=False)
   df.swap_dims({'depth': 'sig0'}).prof_CT.plot(ax=ax3, y='sig0', marker='.', yincrease=False)
   df.swap_dims({'depth': 'sig0'}).prof_SA.plot(ax=ax4, y='sig0', marker='.', yincrease=False)
   #fig.subplots_adjust(wspace=0.7)
   # save figure and close
   plt.savefig(ploc + 'single_profile.png', bbox_inches='tight')
   plt.savefig(ploc + 'single_profile.pdf', bbox_inches='tight')
   plt.show()
   plt.close()

#####################################################################
# Plot many profiles
#####################################################################
def plot_many_profiles(ploc, df, frac = 0.10,
                       zmin = 20, zmax = 1000,
                       Tmin = -1.9, Tmax = 7.0,
                       Smin = 33.0, Smax = 35.0,
                       sig0min = 23.0, sig0max = 28.0,
                       alpha = 0.05, modStr = '',
                       colorVal = 'black'):

   print("plot_tools.plot_many_profiles")

   # font size
   fs = 14

   # p = 20 dbar
   dploc = ploc + 'profile_stats/'
   if not os.path.exists(dploc):
       os.makedirs(dploc)

   # number of profiles
   Nprof = df.profile.values.size

   # select random samples
   sample_size = int(frac*df.profile.size)
   rows_id = sorted(random.sample(range(0, df.profile.size-1), sample_size))
   df_sample = df.isel(profile=rows_id)

   # extract DataArrays
   z = df_sample.depth.values
   depth_highz = df_sample.depth_highz.values
   sig0_levs = df_sample.sig0_levs.values
   CT = df_sample.prof_CT.values
   SA = df_sample.prof_SA.values
   sig0 = df_sample.sig0.values
   sig0_on_highz = df_sample.sig0_on_highz.values
   CTsig = df_sample.ct_on_sig0.values
   SAsig = df_sample.sa_on_sig0.values

   # Rechunk into a single dask array chunk along the "profile" dimension
   # --- this was necessary to get rid of a "core dimension" error
   df = df.chunk(dict(profile=-1))

   # 0.25 quantile
   CT_q25 = df.prof_CT.quantile(0.25, dim='profile').values
   SA_q25 = df.prof_SA.quantile(0.25, dim='profile').values
   sig0_q25 = df.sig0.quantile(0.25, dim='profile').values
   CTsig_q25 = df.ct_on_sig0.quantile(0.25, dim='profile').values
   SAsig_q25 = df.sa_on_sig0.quantile(0.25, dim='profile').values

   # median values
   CT_median = df.prof_CT.quantile(0.50, dim='profile').values
   SA_median = df.prof_SA.quantile(0.50, dim='profile').values
   sig0_median = df.sig0.quantile(0.50, dim='profile').values
   CTsig_median = df.ct_on_sig0.quantile(0.50, dim='profile').values
   SAsig_median = df.sa_on_sig0.quantile(0.50, dim='profile').values

   # 0.75 quantile
   CT_q75 = df.prof_CT.quantile(0.75, dim='profile').values
   SA_q75 = df.prof_SA.quantile(0.75, dim='profile').values
   sig0_q75 = df.sig0.quantile(0.75, dim='profile').values
   CTsig_q75 = df.ct_on_sig0.quantile(0.75, dim='profile').values
   SAsig_q75 = df.sa_on_sig0.quantile(0.75, dim='profile').values

   # figure CT
   fig1, ax1 = plt.subplots()
   for d in range(CT.shape[0]):
       ax1.plot(CT[d,:], z, lw = 1, alpha = alpha, color = 'grey')

   ax1.plot(CT_q25, z, lw = 2, color = colorVal, linestyle='dashed')
   ax1.plot(CT_median, z, lw = 2, color = colorVal)
   ax1.plot(CT_q75, z, lw = 2, color = colorVal, linestyle='dashed')
   ax1.set_xlim([Tmin, Tmax])
   ax1.set_ylim([zmin, zmax])
   plt.gca().invert_yaxis()
   plt.xlabel('Conservative temperature (°C)', fontsize=fs)
   plt.ylabel('Depth (m)', fontsize=fs)
   ax1.tick_params(axis='x', labelsize=fs)
   ax1.tick_params(axis='y', labelsize=fs)
   #plt.text(0.1, 0.1, modStr, ha='left', va='bottom', fontsize=fs, transform=ax1.transAxes)
   #plt.title(modStr, fontsize=fs)
   plt.text(0.9, 0.1, 'N = ' + str(Nprof), \
            ha='right', va='bottom', fontsize=fs, transform=ax1.transAxes)
   plt.savefig(dploc + modStr + 'many_profiles_CT.png', bbox_inches='tight')
   plt.savefig(dploc + modStr + 'many_profiles_CT.pdf', bbox_inches='tight')
   plt.show()
   plt.close()

   # figure SA
   fig1, ax1 = plt.subplots()
   for d in range(SA.shape[0]):
       ax1.plot(SA[d,:], z, lw = 1, alpha = alpha, color = 'grey')

   ax1.plot(SA_q25, z, lw = 2, color = colorVal, linestyle='dashed')
   ax1.plot(SA_median, z, lw = 2, color = colorVal)
   ax1.plot(SA_q75, z, lw = 2, color = colorVal, linestyle='dashed')
   ax1.set_xlim([Smin, Smax])
   ax1.set_ylim([zmin, zmax])
   plt.gca().invert_yaxis()
   plt.xlabel('Absolute salinity (psu)', fontsize=fs)
   plt.ylabel('Depth (m)', fontsize=fs)
   ax1.tick_params(axis='x', labelsize=fs)
   ax1.tick_params(axis='y', labelsize=fs)
   #plt.text(0.1, 0.1, modStr, ha='left', va='bottom', fontsize=fs, transform=ax1.transAxes)
   #plt.title(modStr, fontsize=fs)
   plt.text(0.9, 0.1, 'N = ' + str(Nprof), \
            ha='right', va='bottom', fontsize=fs, transform=ax1.transAxes)
   plt.savefig(dploc + modStr + 'many_profiles_SA.png', bbox_inches='tight')
   plt.savefig(dploc + modStr + 'many_profiles_SA.pdf', bbox_inches='tight')
   plt.show()
   plt.close()

   # figure sig0
   fig1, ax1 = plt.subplots()
   for d in range(sig0.shape[0]):
       ax1.plot(sig0[d,:], z, lw = 1, alpha = alpha, color = 'grey')

   ax1.plot(sig0_q25, z, lw = 2, color = colorVal, linestyle='dashed')
   ax1.plot(sig0_median, z, lw = 2, color = colorVal)
   ax1.plot(sig0_q75, z, lw = 2, color = colorVal, linestyle='dashed')
   ax1.set_xlim([sig0min, sig0max])
   ax1.set_ylim([zmin, zmax])
   plt.gca().invert_yaxis()
   plt.xlabel('Potential density (kg/m^3)', fontsize=fs)
   plt.ylabel('Depth (m)', fontsize=fs)
   ax1.tick_params(axis='x', labelsize=fs)
   ax1.tick_params(axis='y', labelsize=fs)
   #plt.text(0.1, 0.1, modStr, ha='left', va='bottom', fontsize=fs, transform=ax1.transAxes)
   #plt.title(modStr, fontsize=fs)
   plt.text(0.9, 0.1, 'N = ' + str(Nprof), \
            ha='right', va='bottom', fontsize=fs, transform=ax1.transAxes)
   plt.savefig(dploc + modStr + 'many_profiles_sig0.png', bbox_inches='tight')
   plt.savefig(dploc + modStr + 'many_profiles_sig0.pdf', bbox_inches='tight')
   plt.show()
   plt.close()

   # figure CT sig
   fig1, ax1 = plt.subplots()
   for d in range(CTsig.shape[0]):
       ax1.plot(CTsig[d,:], sig0_levs, lw = 1, alpha = alpha, color = 'grey')

   ax1.plot(CTsig_q25, sig0_levs, lw = 2, color = colorVal, linestyle='dashed')
   ax1.plot(CTsig_median, sig0_levs, lw = 2, color = colorVal)
   ax1.plot(CTsig_q75, sig0_levs, lw = 2, color = colorVal, linestyle='dashed')
   ax1.set_xlim([Tmin, Tmax])
   ax1.set_ylim([sig0min, sig0max])
   plt.gca().invert_yaxis()
   plt.xlabel('Conservative temperature (°C)', fontsize=fs)
   plt.ylabel('Potential density (kg/m^3)', fontsize=fs)
   ax1.tick_params(axis='x', labelsize=fs)
   ax1.tick_params(axis='y', labelsize=fs)
   #plt.text(0.1, 0.1, modStr, ha='left', va='bottom', fontsize=fs, transform=ax1.transAxes)
   #plt.title(modStr, fontsize=fs)
   plt.text(0.9, 0.1, 'N = ' + str(Nprof), \
            ha='right', va='bottom', fontsize=fs, transform=ax1.transAxes)
   plt.savefig(dploc + modStr + 'many_profiles_CTsig.png', bbox_inches='tight')
   plt.savefig(dploc + modStr + 'many_profiles_CTsig.pdf', bbox_inches='tight')
   plt.show()
   plt.close()

   # figure SA sig
   fig1, ax1 = plt.subplots()
   for d in range(SAsig.shape[0]):
       ax1.plot(SAsig[d,:], sig0_levs, lw = 1, alpha = alpha, color = 'grey')

   ax1.plot(SAsig_q25, sig0_levs, lw = 2, color = colorVal, linestyle='dashed')
   ax1.plot(SAsig_median, sig0_levs, lw = 2, color = colorVal)
   ax1.plot(SAsig_q75, sig0_levs, lw = 2, color = colorVal, linestyle='dashed')
   ax1.set_xlim([Smin, Smax])
   ax1.set_ylim([sig0min, sig0max])
   plt.gca().invert_yaxis()
   plt.xlabel('Absolute salinity (psu)', fontsize=fs)
   plt.ylabel('Potential density (kg/m^3)', fontsize=fs)
   ax1.tick_params(axis='x', labelsize=fs)
   ax1.tick_params(axis='y', labelsize=fs)
   #plt.text(0.1, 0.1, modStr, ha='left', va='bottom', fontsize=fs, transform=ax1.transAxes)
   #plt.title(modStr, fontsize=fs)
   plt.text(0.9, 0.1, 'N = ' + str(Nprof), \
            ha='right', va='bottom', fontsize=fs, transform=ax1.transAxes)
   plt.savefig(dploc + modStr + 'many_profiles_SAsig.pdf', bbox_inches='tight')
   plt.savefig(dploc + modStr + 'many_profiles_SAsig.png', bbox_inches='tight')
   plt.show()
   plt.close()

#####################################################################
# Plot single profile
#####################################################################
def plot_profiles_on_density_levels(ploc, profiles, frac=0.33):

    # start message
    print('plot_tools.plot_profiles_on_density_levels')

    # subdirectory
    dploc = ploc + 'profile_stats/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)

    # conservative temp
    plt.figure(figsize=(30, 30))
    profiles.ct_on_sig0.plot()
    plt.savefig(dploc + 'ct_on_sig0.pdf', bbox_inches='tight')
    plt.savefig(dploc + 'ct_on_sig0.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # absolute salinity
    plt.figure(figsize=(30, 30))
    profiles.sa_on_sig0.plot()
    plt.savefig(dploc + 'sa_on_sig0.pdf', bbox_inches='tight')
    plt.savefig(dploc + 'sa_on_sig0.png', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot PCA vertical structure
#####################################################################
def plot_pca_vertical_structure(ploc, profiles, pca, Xpca):

    # start message
    print('plot_tools.plot_pca')

    # split into temp (first half) and salt (second half)
    y = pca.components_
    n = int(y.shape[1])

    # condition to check length is EVEN or not
    # if lenght is ODD, show message and exit
    if( n%2 != 0 ):
        print('plot_tools.plot_pca: bad PCA array')
        exit()

    # pca temp and salinity
    pca_temp = y[:, 0:int(n/2)]
    pca_salt = y[:, int(n/2):n]

    ############# Temperature plot

    # initialize the figure
    plt.figure(figsize=(30, 30))
    plt.style.use('seaborn-darkgrid')
    #palette = cmx.Paired(np.linspace(0,1,n_comp))

    # vertical coordinate
    z = profiles.depth.values

    # iterate over groups
    num = 0
    for npca in range(pca.n_components):
        num += 1

        # select subplot
        ax = plt.subplot(5,3,num)
        plt.plot(pca_temp[npca,:], z, marker='', linestyle='solid',
                 color='black', linewidth=6.0, alpha=0.9)

        # custom grid and axes
        plt.ylim([min(z), max(z)])
        #plt.xlim([33.6, 37.0])

        #text box
        fs = 42 # font size
        plt.xlabel('PC', fontsize=fs)
        plt.ylabel('Depth (m)', fontsize=fs)
        plt.title('PC' + str(num) + ' (temp)', fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    # save figure and close
    plt.savefig(ploc + 'pca_temp.png', bbox_inches='tight')
    plt.savefig(ploc + 'pca_temp.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    ############# Salinity plot

    # initialize the figure
    plt.figure(figsize=(30, 30))
    plt.style.use('seaborn-darkgrid')
    #palette = cmx.Paired(np.linspace(0,1,n_comp))

    # vertical coordinate
    z = profiles.depth.values

    # iterate over groups
    num = 0
    for npca in range(pca.n_components):
        num += 1

        # select subplot
        ax = plt.subplot(5,3,num)
        plt.plot(pca_salt[npca,:], z, marker='', linestyle='solid',
                 color='black', linewidth=6.0, alpha=0.9)

        # custom grid and axes
        plt.ylim([min(z), max(z)])
        #plt.xlim([33.6, 37.0])

        #text box
        fs = 42 # font size
        plt.xlabel('PC', fontsize=fs)
        plt.ylabel('Depth (m)', fontsize=fs)
        plt.title('PC' + str(num) + ' (salt)', fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    # save figure and close
    plt.savefig(ploc + 'pca_salt.png', bbox_inches='tight')
    plt.savefig(ploc + 'pca_salt.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot PCA structure in 3D space (with or without class labels
#####################################################################
def plot_pca3D(ploc, colormap, profiles, Xpca, frac=0.33, withLabels=False):

    # for the ellipses
    import numpy.linalg as la

    # start message
    print('plot_tools.plot_pca3D')

    # just to shorten the names
    xy=Xpca

    # if plot with label
    if withLabels==True:
        labels=profiles.label.values

    # random sample
    rsample_size = int(frac*xy.shape[0])
    rows_id = random.sample(range(0,xy.shape[0]-1), rsample_size)

    # select radom sample in xy and color
    xyp = xy[rows_id,:]

    if withLabels==True:
        c = labels[rows_id]
        labs='labelled'
    else:
        c = [[ 0.267004,  0.004874,  0.329415,  1.  ]]
        labs='nolabels'

    # 3D scatterplots\
    # ax.set_xlim()  use these to zoom in a bit

    qlow = 0.001
    qhi = 0.999

    # view 1
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, 0)
    CS = ax.scatter(xyp[:,0], xyp[:,1], xyp[:,2], 'o', c=c, cmap=colormap, alpha=1.0, s=1.5)
    ax.set_xlim(np.round(np.quantile(xyp[:,0], qlow)), np.round(np.quantile(xyp[:,0], qhi)))
    ax.set_ylim(np.round(np.quantile(xyp[:,1], qlow)), np.round(np.quantile(xyp[:,1], qhi)))
    ax.set_zlim(np.round(np.quantile(xyp[:,2], qlow)), np.round(np.quantile(xyp[:,2], qhi)))
    plt.savefig(ploc + 'pca_scatter_' + labs + '_view1' + '.png', bbox_inches='tight')
    plt.savefig(ploc + 'pca_scatter_' + labs + '_view1' + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # view 2
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, 120)
    CS = ax.scatter(xyp[:,0], xyp[:,1], xyp[:,2], 'o', c=c, cmap=colormap, alpha=1.0, s=1.5)
    ax.set_xlim(np.round(np.quantile(xyp[:,0], qlow)), np.round(np.quantile(xyp[:,0], qhi)))
    ax.set_ylim(np.round(np.quantile(xyp[:,1], qlow)), np.round(np.quantile(xyp[:,1], qhi)))
    ax.set_zlim(np.round(np.quantile(xyp[:,2], qlow)), np.round(np.quantile(xyp[:,2], qhi)))
    plt.savefig(ploc + 'pca_scatter_' + labs + '_view2' + '.png', bbox_inches='tight')
    plt.savefig(ploc + 'pca_scatter_' + labs + '_view2' + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # view 3
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, 240)
    CS = ax.scatter(xyp[:,0], xyp[:,1], xyp[:,2], 'o', c=c, cmap=colormap, alpha=1.0, s=1.5)
    ax.set_xlim(np.round(np.quantile(xyp[:,0], qlow)), np.round(np.quantile(xyp[:,0], qhi)))
    ax.set_ylim(np.round(np.quantile(xyp[:,1], qlow)), np.round(np.quantile(xyp[:,1], qhi)))
    ax.set_zlim(np.round(np.quantile(xyp[:,2], qlow)), np.round(np.quantile(xyp[:,2], qhi)))
    plt.savefig(ploc + 'pca_scatter_' + labs + '_view3' + '.png', bbox_inches='tight')
    plt.savefig(ploc + 'pca_scatter_' + labs + '_view3' + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # separate colorbar (to be added)

#####################################################################
# Pairplot (general)
#####################################################################
def plot_pairs(ploc, dataset, kind="hist", descr=""):

    # start message
    print('plot_tools.plot_pairs')

    # create pandas dataframe from numpy array
    df = pd.DataFrame(data=dataset)

    # create figure
    fig = plt.figure(figsize=(15,15))
    sns.pairplot(df, kind=kind)
    plt.savefig(ploc + 'pairplot_' + descr + '.png', bbox_inches='tight')
    plt.savefig(ploc + 'pairplot_' + descr + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot UMAP structures (not shaded by class or anything yet)
#####################################################################
def plot_umap(ploc, Xtrans, frac=0.33):

    # start message
    print('plot_tools.plot_umap')

    # just to shorten the name
    xy=Xtrans
    # random sample
    rsample_size = int(frac*xy.shape[0])
    rows_id = random.sample(range(0,xy.shape[0]-1), rsample_size)
    xyp = xy[rows_id,:]

    # view 1
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, 0)
    ax.scatter(xyp[:,0], xyp[:,1], xyp[:,2], 'o', s=1.0)
    plt.savefig(ploc + 'umap_scatter_nolabels_view1' + '.png', bbox_inches='tight')
    plt.savefig(ploc + 'umap_scatter_nolabels_view1' + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # view 2
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, 120)
    ax.scatter(xyp[:,0], xyp[:,1], xyp[:,2], 'o', s=1.0)
    plt.savefig(ploc + 'umap_scatter_nolabels_view2' + '.png', bbox_inches='tight')
    plt.savefig(ploc + 'umap_scatter_nolabels_view2' + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # view 3
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, 240)
    ax.scatter(xyp[:,0], xyp[:,1], xyp[:,2], 'o', s=1.0)
    plt.savefig(ploc + 'umap_scatter_nolabels_view3' + '.png', bbox_inches='tight')
    plt.savefig(ploc + 'umap_scatter_nolabels_view3' + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot BIC scores
#####################################################################
def plot_bic_scores(ploc, max_N, bic_mean, bic_std):

    # start message
    print('plot_tools.plot_bic_scores')

    # plot the BIC scores
    n_components_range = range(2, max_N)
    plt.figure(figsize=(20, 8))
    spl = plt.subplot(2, 1, 1)
    plt.plot(n_components_range, bic_mean-bic_std, '--', color='black')
    plt.plot(n_components_range, bic_mean, '-', color='black')
    plt.plot(n_components_range, bic_mean+bic_std, '--', color='black')
    plt.xticks(n_components_range)
    plt.title('BIC score per model', fontsize=18)
    spl.set_xlabel('Number of components',fontsize=18)
    spl.set_ylabel('BIC score',fontsize=18)
    # save figure
    plt.savefig(ploc + 'bic_scores.png', bbox_inches='tight')
    plt.savefig(ploc + 'bic_scores.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # calculate and plot delta_bic
    delta_bic = np.diff(bic_mean)   # out[i] = a[i+1] - a[i]
    plt.figure(figsize=(20,8))
    spl = plt.subplot(2,1,1)
    plt.plot(n_components_range[:-1], delta_bic, '-', color='black')
    plt.xticks(n_components_range[:-1])
    plt.title('Change in BIC score per model', fontsize=18)
    spl.set_xlabel('Number of components',fontsize=18)
    spl.set_ylabel('Change in BIC score',fontsize=18)
    # save figure
    plt.savefig(ploc + 'delta_bic_scores.png', bbox_inches='tight')
    plt.savefig(ploc + 'delta_bic_scores.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot AIC scores
#####################################################################
def plot_aic_scores(ploc, max_N, aic_mean, aic_std):

    # start message
    print('plot_tools.plot_aic_scores')

    # plot the AIC scores
    n_components_range = range(2, max_N)
    plt.figure(figsize=(20, 8))
    spl = plt.subplot(2, 1, 1)
    plt.plot(n_components_range, aic_mean-aic_std, '--', color='black')
    plt.plot(n_components_range, aic_mean, '-', color='black')
    plt.plot(n_components_range, aic_mean+aic_std, '--', color='black')
    plt.xticks(n_components_range)
    plt.title('AIC score per model', fontsize=18)
    spl.set_xlabel('Number of components',fontsize=18)
    spl.set_ylabel('AIC score',fontsize=18)
    # save figure
    plt.savefig(ploc + 'aic_scores.png', bbox_inches='tight')
    plt.savefig(ploc + 'aic_scores.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # calculate and plot delta_aic
    delta_aic = np.diff(aic_mean)   # out[i] = a[i+1] - a[i]
    plt.figure(figsize=(20,8))
    spl = plt.subplot(2,1,1)
    plt.plot(n_components_range[:-1], delta_aic, '-', color='black')
    plt.xticks(n_components_range[:-1])
    plt.title('Change in AIC score per model', fontsize=18)
    spl.set_xlabel('Number of components',fontsize=18)
    spl.set_ylabel('Change in AIC score',fontsize=18)
    # save figure
    plt.savefig(ploc + 'delta_aic_scores.png', bbox_inches='tight')
    plt.savefig(ploc + 'delta_aic_scores.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Sample profile plots for T, S, CT, SA, and sig0
#####################################################################
def prof_TS_sample_plots(ploc, profiles):

    # start message
    print('plot_tools.plot_TS_sample_plots')

    # subdirectory
    dploc = ploc + 'sample_plot_and_hist/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)

    # plotting subset (just a few for visualisation purposes)
    subset = range(1000,2000,1)
    # temperature plots
    fig, ax = plt.subplots(figsize=(15,10))
    profiles.prof_T[subset].plot(y='depth', yincrease=False)
    plt.savefig(dploc + 'prof_T_subset.png', bbox_inches='tight')
    plt.savefig(dploc + 'prof_T_subset.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    # salinity plots
    fig, ax = plt.subplots(figsize=(15,10))
    profiles.prof_S[subset].plot(y='depth', yincrease=False)
    plt.savefig(dploc + 'prof_S_subset.png', bbox_inches='tight')
    plt.savefig(dploc + 'prof_S_subset.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    # conservative temperature plots
    fig, ax = plt.subplots(figsize=(15,10))
    profiles.prof_CT[subset].plot(y='depth', yincrease=False)
    plt.savefig(dploc + 'prof_CT_subset.png', bbox_inches='tight')
    plt.savefig(dploc + 'prof_CT_subset.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    # absolute salinity plots
    fig, ax = plt.subplots(figsize=(15,10))
    profiles.prof_SA[subset].plot(y='depth', yincrease=False)
    plt.savefig(dploc + 'prof_SA_subset.png', bbox_inches='tight')
    plt.savefig(dploc + 'prof_SA_subset.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    # sigma0 density
    fig, ax = plt.subplots(figsize=(15,10))
    profiles.sig0[subset].plot(y='depth', yincrease=False)
    plt.savefig(dploc + 'sig0_subset.png', bbox_inches='tight')
    plt.savefig(dploc + 'sig0_subset.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    # histograms
    xr.plot.hist(profiles.prof_T,figsize=(15,10))
    plt.savefig(dploc + 'hist_prof_T.png', bbox_inches='tight')
    plt.savefig(dploc + 'hist_prof_T.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    xr.plot.hist(profiles.prof_S,figsize=(15,10))
    plt.savefig(dploc + 'hist_prof_S.png', bbox_inches='tight')
    plt.savefig(dploc + 'hist_prof_S.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    xr.plot.hist(profiles.prof_CT,figsize=(15,10))
    plt.savefig(dploc + 'hist_prof_CT.png', bbox_inches='tight')
    plt.savefig(dploc + 'hist_prof_CT.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    xr.plot.hist(profiles.prof_SA,figsize=(15,10))
    plt.savefig(dploc + 'hist_prof_SA.png', bbox_inches='tight')
    plt.savefig(dploc + 'hist_prof_SA.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
    xr.plot.hist(profiles.sig0,figsize=(15,10))
    plt.savefig(dploc + 'hist_sig0.png', bbox_inches='tight')
    plt.savefig(dploc + 'hist_sig0.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot PCA structure for temperature and salinity
#####################################################################
def plot_pca_structure(ploc, profiles, pca, number_of_pca_components, zmin, zmax):

    # start message
    print('plot_tools.plot_pca_structure')

    # ------- TEMPERATURE PCA COMPONENTS
    # initialize the figure
    plt.figure(figsize=(30, 30))
    plt.style.use('seaborn-darkgrid')
    #palette = cmx.Paired(np.linspace(0,1,n_comp))

    # vertical coordinate
    z = profiles.depth.values

    # iterate over groups
    num = 0
    for npca in range(number_of_pca_components):
        num += 1

        # select subplot
        ax = plt.subplot(2,3,num)
        plt.plot(pca.components_[npca,0:15], z, marker='', linestyle='solid',
                 color='black', linewidth=6.0, alpha=0.9)

        # custom grid and axes
        plt.ylim([zmin, zmax])
        #plt.xlim([33.6, 37.0])

        #text box
        fs = 42 # font size
        plt.xlabel('PC', fontsize=fs)
        plt.ylabel('Depth (m)', fontsize=fs)
        plt.title('PC' + str(num) + ' (CT)', fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    plt.savefig(ploc + 'pca_CT.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # ------- SALINITY PCA COMPONENTS
    # initialize the figure
    plt.figure(figsize=(30, 30))
    plt.style.use('seaborn-darkgrid')
    #palette = cmx.Paired(np.linspace(0,1,n_comp))

    # vertical coordinate
    z = profiles.depth.values

    # iterate over groups
    num = 0
    for npca in range(number_of_pca_components):
        num += 1

        # select subplot
        ax = plt.subplot(2,3,num)
        plt.plot(pca.components_[npca,15:], z, marker='', linestyle='solid',
                 color='black', linewidth=6.0, alpha=0.9)

        # custom grid and axes
        plt.ylim([zmin, zmax])
        #plt.xlim([33.6, 37.0])

        #text box
        fs = 42 # font size
        plt.xlabel('PC', fontsize=fs)
        plt.ylabel('Depth (m)', fontsize=fs)
        plt.title('PC' + str(num) + ' (SA)', fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    plt.savefig(ploc + 'pca_SA.png', bbox_inches='tight')
    plt.savefig(ploc + 'pca_SA.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot mean and stdev CT and SA structure
#####################################################################
def plot_CT_and_SA_class_structure(ploc, profiles, class_means,
                                   class_stds, n_components_selected,
                                   colormap,
                                   zmin=20, zmax=1000,
                                   Tmin=-3, Tmax=20,
                                   Smin=33.6, Smax=37.0,
                                   sig0min=26.0, sig0max=28.0, frac=0.33):

    print('plot_tools.N')

    # select colormap
    #colormap = plt.get_cmap('Set1', n_components_selected)
    cNorm = colors.Normalize(vmin=0, vmax=n_components_selected)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)

    # initialize the figure
    fig = plt.figure(figsize=(22,16))

    # extract DataArrays
    z = profiles.depth.values

    fs = 18 # font size

    # iterate over groups (top row, CT)
    num = 0
    for nrow in range(0,n_components_selected):
        num += 1
        colorVal = scalarMap.to_rgba(nrow)

        # extract means
        mean_T = class_means.prof_CT[nrow,:].values
        # extract stdevs
        std_T = class_stds.prof_CT[nrow,:].values

        # select subplot
        ax = plt.subplot(2,n_components_selected,num)
        plt.plot(mean_T, z, marker='', linestyle='solid', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_T+std_T, z, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_T-std_T, z, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)

        # custom grid and axes
        plt.ylim([zmin,zmax])
        plt.xlim([Tmin, Tmax])

        #text box
        if nrow==0:
            #plt.xlabel('Conservative temperature (deg C)', fontsize=fs)
            plt.ylabel('Depth (m)', fontsize=fs)

        #plt.title('Class = ' + str(num), fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=fs)
        ax.tick_params(axis='y', labelsize=fs)

    # iterate over groups (SA)
    for nrow in range(0,n_components_selected):
        num += 1
        colorVal = scalarMap.to_rgba(nrow)

        # extract means
        mean_S = class_means.prof_SA[nrow,:].values
        # extract stdevs
        std_S = class_stds.prof_SA[nrow,:].values

        # select subplot
        ax = plt.subplot(2,n_components_selected,num)
        plt.plot(mean_S, z, marker='', linestyle='solid', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_S+std_S, z, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_S-std_S, z, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)

        # custom grid and axes
        plt.ylim([zmin, zmax])
        plt.xlim([Smin, Smax])

        #text box
        if nrow==0:
            #plt.xlabel('Conservative temperature (deg C)', fontsize=fs)
            plt.ylabel('Depth (m)', fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=fs)
        ax.tick_params(axis='y', labelsize=fs)

    #fig.subplots_adjust(wspace=0.7)
    plt.tight_layout(pad=3.0)
    plt.savefig(ploc + 'prof_CT_and_SA_byClass.png', bbox_inches='tight')
    plt.savefig(ploc + 'prof_CT_and_SA_byClass.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot vertical structure of a single class (CT, SA, sigma0)
#####################################################################
def plot_class_vertical_structures(ploc, df1, n_components_selected, colormap,
                                   zmin=20, zmax=1000,
                                   Tmin=-3, Tmax=20,
                                   Smin=33.6, Smax=37.0,
                                   sig0min=26.0, sig0max=28.0,
                                   frac=0.33, description='full'):
# note: the input 'df1' should contain only a single class/label!

    print('plot_tools.plot_class_vertical_structures')

    # select colormap
    #colormap = plt.get_cmap('Set1', n_components_selected)
    cNorm = colors.Normalize(vmin=0, vmax=n_components_selected)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)

    # iterate over groups (top row, CT)
    for nrow in range(0,n_components_selected):
        colorVal = scalarMap.to_rgba(nrow)

        # just select single class/label
        df1_singleClass = df1.where(df1.label==nrow, drop=True)

        # call routine to plot many profiles
        plot_many_profiles(ploc, df1_singleClass, frac=0.10,
                           zmin=zmin, zmax=zmax,
                           Tmin=Tmin, Tmax=Tmax,
                           Smin=Smin, Smax=Smax,
                           sig0min=sig0min, sig0max=sig0max,
                           alpha=0.05,
                           modStr='Class'+str(nrow)+'z'+description,
                           colorVal=colorVal)

#####################################################################
# Plot mean and stdev salinity class structure
#####################################################################
def plot_SA_class_structure(ploc, profiles, class_means,
                           class_stds, n_components_selected, colormap,
                           zmin, zmax, Smin=33.6, Smax=37.0):

    print('plot_tools.plot_SA_class_structure')

    # select colormap
    #colormap = plt.get_cmap('Set1', n_components_selected)
    cNorm = colors.Normalize(vmin=0, vmax=n_components_selected)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)

    # initialize the figure
    fig = plt.figure(figsize=(60, 100))
    #plt.style.use('seaborn-darkgrid')
    #palette = cmx.Paired(np.linspace(0,1,n_comp))

    # vertical coordinate
    z = profiles.depth.values

    # iterate over groups
    num = 0
    for nrow in range(0,n_components_selected):
        num += 1
        colorVal = scalarMap.to_rgba(nrow)

        # extract means
        mean_S = class_means.prof_SA[nrow,:].values

        # extract stdevs
        std_S = class_stds.prof_SA[nrow,:].values

        # select subplot
        ax = plt.subplot(7,2,num)
        plt.plot(mean_S, z, marker='', linestyle='solid', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_S+std_S, z, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_S-std_S, z, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)

        # custom grid and axes
        plt.ylim([zmin, zmax])
        plt.xlim([Smin, Smax])

       #text box
        fs = 42 # font size
        plt.xlabel('Absolute salinity (psu)', fontsize=fs)
        plt.ylabel('Depth (m)', fontsize=fs)
        #plt.title('Class = ' + str(num), fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    #fig.subplots_adjust(wspace=0.7)
    plt.savefig(ploc + 'prof_SA_byClass.png', bbox_inches='tight')
    plt.savefig(ploc + 'prof_SA_byClass.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot mean and stdev conservative temperature class structure
#####################################################################
def plot_CT_class_structure(ploc, profiles, class_means,
                            class_stds, n_components_selected, colormap,
                            zmin, zmax, Tmin=-3, Tmax=20):

    print('plot_tools.plot_CT_class_structure')

    # select colormap
    #colormap = plt.get_cmap('Set1', n_components_selected)
    cNorm = colors.Normalize(vmin=0, vmax=n_components_selected)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)

    # initialize the figure
    fig = plt.figure(figsize=(60, 100))
    #plt.style.use('seaborn-darkgrid')
    #palette = cmx.Paired(np.linspace(0,1,n_comp))

    # vertical coordinate
    z = profiles.depth.values

    # iterate over groups
    num = 0
    for nrow in range(0,n_components_selected):
        num += 1
        colorVal = scalarMap.to_rgba(nrow)

        # extract means
        mean_T = class_means.prof_CT[nrow,:].values

        # extract stdevs
        std_T = class_stds.prof_CT[nrow,:].values

        # select subplot
        ax = plt.subplot(7,2,num)
        plt.plot(mean_T, z, marker='', linestyle='solid', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_T+std_T, z, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_T-std_T, z, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)

        # custom grid and axes
        plt.ylim([zmin,zmax])
        plt.xlim([Tmin, Tmax])

        #text box
        fs = 42 # font size
        plt.xlabel('Conservative temperature (deg C)', fontsize=fs)
        plt.ylabel('Depth (m)', fontsize=fs)
        #plt.title('Class = ' + str(num), fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    #fig.subplots_adjust(wspace=0.7)
    plt.savefig(ploc + 'prof_CT_byClass.png', bbox_inches='tight')
    plt.savefig(ploc + 'prof_CT_byClass.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot mean and stdev salinity class structure (on sigma)
#####################################################################
def plot_SA_class_structure_onSig(ploc, profiles, class_means,
                           class_stds, n_components_selected, colormap,
                           Smin=33.6, Smax=37.0):

    print('plot_tools.plot_SA_class_structure_onSig')

    # should make LaTeX rendering possible
    #plt.rcParams['text.usetex'] = True

    # select colormap
    #colormap = plt.get_cmap('Set1', n_components_selected)
    cNorm = colors.Normalize(vmin=0, vmax=n_components_selected)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)

    # initialize the figure
    fig = plt.figure(figsize=(60, 100))
    #plt.style.use('seaborn-darkgrid')
    #palette = cmx.Paired(np.linspace(0,1,n_comp))

    # vertical coordinate
    sig0 = profiles.sig0_levs.values

    # iterate over groups
    num = 0
    for nrow in range(0,n_components_selected):
        num += 1
        colorVal = scalarMap.to_rgba(nrow)

        # extract means
        mean_S = class_means.sa_on_sig0[nrow,:].values

        # extract stdevs
        std_S = class_stds.sa_on_sig0[nrow,:].values

        # select subplot
        ax = plt.subplot(7,2,num)
        plt.plot(mean_S, sig0, marker='', linestyle='solid', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_S+std_S, sig0, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_S-std_S, sig0, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)

        # custom grid and axes
        #plt.ylim([zmin, zmax])
        plt.xlim([Smin, Smax])

       #text box
        fs = 42 # font size
        plt.xlabel('Absolute salinity (psu)', fontsize=fs)
        plt.ylabel('Potential density (kg/m^3)', fontsize=fs)
        #plt.title('Class = ' + str(num), fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    #fig.subplots_adjust(wspace=0.7)
    plt.savefig(ploc + 'prof_SA_sig0_byClass.png', bbox_inches='tight')
    plt.savefig(ploc + 'prof_SA_sig0_byClass.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot mean and stdev conservative temperature class structure
#####################################################################
def plot_CT_class_structure_onSig(ploc, profiles, class_means,
                            class_stds, n_components_selected, colormap,
                            Tmin=-3, Tmax=20):

    print('plot_tools.plot_CT_class_structure_onSig')

    # should make LaTeX rendering possible
    #plt.rcParams['text.usetex'] = True

    # select colormap
    #colormap = plt.get_cmap('Set1', n_components_selected)
    cNorm = colors.Normalize(vmin=0, vmax=n_components_selected)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)

    # initialize the figure
    fig = plt.figure(figsize=(60, 100))
    #plt.style.use('seaborn-darkgrid')
    #palette = cmx.Paired(np.linspace(0,1,n_comp))

    # vertical coordinate
    sig0 = profiles.sig0_levs.values

    # iterate over groups
    num = 0
    for nrow in range(0,n_components_selected):
        num += 1
        colorVal = scalarMap.to_rgba(nrow)

        # extract means
        mean_T = class_means.ct_on_sig0[nrow,:].values

        # extract stdevs
        std_T = class_stds.ct_on_sig0[nrow,:].values

        # select subplot
        ax = plt.subplot(7,2,num)
        plt.plot(mean_T, sig0, marker='', linestyle='solid', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_T+std_T, sig0, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_T-std_T, sig0, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)

        # custom grid and axes
        #plt.ylim([zmin,zmax])
        plt.xlim([Tmin, Tmax])

        #text box
        fs = 42 # font size
        plt.xlabel('Conservative temperature (deg C)', fontsize=fs)
        plt.ylabel('Potential density (kg/m^3)', fontsize=fs)
        #plt.title('Class = ' + str(num), fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    #fig.subplots_adjust(wspace=0.7)
    plt.savefig(ploc + 'prof_CT_sig0_byClass.png', bbox_inches='tight')
    plt.savefig(ploc + 'prof_CT_sig0_byClass.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot mean and stdev density class structure
#####################################################################
def plot_sig0_class_structure(ploc, profiles, class_means,
                           class_stds, n_components_selected, colormap,
                           zmin, zmax, sig0min=24.0, sig0max=28.0):

    print('plot_tools.plot_sig0_class_structure')

    # should make LaTeX rendering possible
    #plt.rcParams['text.usetex'] = True

    # select colormap
    #colormap = plt.get_cmap('Set1', n_components_selected)
    cNorm = colors.Normalize(vmin=0, vmax=n_components_selected)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)

    # initialize the figure
    fig = plt.figure(figsize=(60, 100))
    #plt.style.use('seaborn-darkgrid')
    #palette = cmx.Paired(np.linspace(0,1,n_comp))

    # vertical coordinate
    z = profiles.depth.values

    # iterate over groups
    num = 0
    for nrow in range(0,n_components_selected):
        num += 1
        colorVal = scalarMap.to_rgba(nrow)

        # extract means
        mean_S = class_means.sig0[nrow,:].values

        # extract stdevs
        std_S = class_stds.sig0[nrow,:].values

        # select subplot
        ax = plt.subplot(7,2,num)
        plt.plot(mean_S, z, marker='', linestyle='solid', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_S+std_S, z, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)
        plt.plot(mean_S-std_S, z, marker='', linestyle='dashed', color=colorVal,
            linewidth=6.0, alpha=0.9)

        # custom grid and axes
        plt.ylim([zmin, zmax])
        plt.xlim([sig0min, sig0max])

       #text box
        fs = 42 # font size
        plt.xlabel('\sigma_0 (kg/m^3)', fontsize=fs)
        plt.ylabel('Depth (m)', fontsize=fs)
        #plt.title('Class = ' + str(num), fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    #fig.subplots_adjust(wspace=0.7)
    plt.savefig(ploc + 'prof_sig0_byClass.png', bbox_inches='tight')
    plt.savefig(ploc + 'prof_sig0_byClass.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Plot class label map using cartopy
#####################################################################
def plot_label_map(ploc, profiles, n_components_selected, colormap,
                   lon_min=-80, lon_max=80, lat_min=-85, lat_max=-30,
                   bathy_fname="bathy.nc", lev_range=range(-6000,1,500)):

    print('plot_tools.plot_label_map')

    # load bathymetry
    bds = io.load_bathymetry(bathy_fname)
    bathy_lon = bds['lon'][:]
    bathy_lat = bds['lat'][:]
    bathy = bds['bathy'][:]

    # define colormap
    #colormap = plt.get_cmap('Set1', n_components_selected)

    # extract values as new DataArrays
    df1D = profiles.isel(depth=0)
    da_lon = df1D.lon
    da_lat = df1D.lat
    da_label = df1D.label

    # extract values
    lons = da_lon.values
    lats = da_lat.values
    clabels = da_label.values

    # size of random sample
    random_sample_size = int(np.ceil(0.30*df1D.profile.size))

    # random sample for plotting
    rows_id = random.sample(range(0,clabels.size-1), random_sample_size)
    lons_random_sample = lons[rows_id]
    lats_random_sample = lats[rows_id]
    clabels_random_sample = clabels[rows_id]

    # scatterplot
    plt.figure(figsize=(17, 13))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())

    # add bathymetry contours
    ax.contour(bathy_lon, bathy_lat, bathy, levels=lev_range,
            linewidths=0.5, alpha=0.5, colors="k", linestyles='-',
            transform=ccrs.PlateCarree())

    # scatter plot
    CS = plt.scatter(lons_random_sample-360,
                     lats_random_sample,
                     c=clabels_random_sample,
                     marker='o',
                     cmap= colormap,
                     s=10.0,
                     transform=ccrs.Geodetic(),
                     )
    ax.coastlines(resolution='50m')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    ax.add_feature(cartopy.feature.LAND)
    #plt.colorbar(CS)

    # save figure
    plt.savefig(ploc + 'label_map.png', bbox_inches='tight')
    plt.savefig(ploc + 'label_map.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Scatter plot single i-metric map
#####################################################################
def plot_i_metric_single_panel(ploc, df1D, lon_min, lon_max, lat_min, lat_max,
        rr=0.66,str="bathy.nc", lev_range=range(-6000,1,500)):

    print('plot_tools.plot_i_metric_single_panel')

    # subdirectory
    dploc = ploc + 'imetric_scatter/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)

    # load bathymetry
    bds = io.load_bathymetry(str)
    bathy_lon = bds['lon'][:]
    bathy_lat = bds['lat'][:]
    bathy = bds['bathy'][:]

    # load fronts
    pf = io.load_front("fronts/pf_kim.txt")
    saccf = io.load_front("fronts/saccf_kim.txt")
    saf = io.load_front("fronts/saf_kim.txt")
    sbdy = io.load_front("fronts/sbdy_kim.txt")

    # extract values as new DataArrays
    da_lon = df1D.lon
    da_lat = df1D.lat
    da_i_metric = df1D.i_metric

    # extract values
    lons = da_lon.values
    lats = da_lat.values
    c = da_i_metric.values

    # size of random sample (all profiles by now)
    random_sample_size = int(np.ceil(rr*df1D.lon.size))

    # random sample for plotting
    rows_id = random.sample(range(0,c.size-1), random_sample_size)
    lons_random_sample = lons[rows_id]
    lats_random_sample = lats[rows_id]
    clabels_random_sample = c[rows_id]

    #colormap with Historical data
    plt.figure(figsize=(17, 13))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())

    # add bathymetry contours
    ax.contour(bathy_lon, bathy_lat, bathy, levels=lev_range,
            linewidths=0.5, alpha=0.5, colors="k", linestyles='-',
            transform=ccrs.PlateCarree())

    # scatter plot
    CS = plt.scatter(lons_random_sample-360,
                     lats_random_sample,
                     c=clabels_random_sample,
                     marker='o',
                     cmap= plt.get_cmap('cividis'),
                     s=10.0,
                     transform=ccrs.Geodetic(),
                     )

    # fronts
    plt.plot(saf[:,0], saf[:,1], color="black", linewidth=2.0, transform=ccrs.Geodetic())
    plt.plot(pf[:,0], pf[:,1], color="blue", linewidth=2.0, transform=ccrs.Geodetic())
    plt.plot(saccf[:,0], saccf[:,1], color="green", linewidth=2.0, transform=ccrs.Geodetic())
    plt.plot(sbdy[:,0], sbdy[:,1], color="yellow", linewidth=2.0, transform=ccrs.Geodetic())

    # coastlines and gridlines
    ax.coastlines(resolution='50m')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
    ax.add_feature(cartopy.feature.LAND)

    # save figure
    plt.savefig(dploc + 'i-metric_single.png', bbox_inches='tight')
    plt.savefig(dploc + 'i-metric_single.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Scatter plot multiple i-metric maps (one per class)
#####################################################################
def plot_i_metric_multiple_panels(ploc, df1D, lon_min, lon_max,
        lat_min, lat_max, n_components_selected, bathy_fname="bathy.nc",
        lev_range=range(-6000,1,500)):

    print('plot_tools.plot_i_metric_multiple_panels')

    # subdirectory
    dploc = ploc + 'imetric_scatter/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)

    # load bathymetry
    bds = io.load_bathymetry(bathy_fname)
    bathy_lon = bds['lon'][:]
    bathy_lat = bds['lat'][:]
    bathy = bds['bathy'][:]

    # load fronts
    pf = io.load_front("fronts/pf_kim.txt")
    saccf = io.load_front("fronts/saccf_kim.txt")
    saf = io.load_front("fronts/saf_kim.txt")
    sbdy = io.load_front("fronts/sbdy_kim.txt")

    # extract values as new DataArrays
    da_lon = df1D.lon
    da_lat = df1D.lat
    da_i_metric = df1D.i_metric
    da_label = df1D.label

    # extract values
    lons = da_lon.values
    lats = da_lat.values
    c = da_i_metric.values
    labels = da_label.values

    for iclass in range(n_components_selected):

        # random sample for plotting
        lons_random_sample = lons[labels==iclass]
        lats_random_sample = lats[labels==iclass]
        clabels_random_sample = c[labels==iclass]

        #colormap with Historical data
        plt.figure(figsize=(17, 13))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())

        # add bathymetry contours
        ax.contour(bathy_lon, bathy_lat, bathy, levels=lev_range,
                linewidths=0.5, alpha=0.5, colors="k", linestyles='-',
                transform=ccrs.PlateCarree())

        # scatter plot
        CS = plt.scatter(lons_random_sample-360,
                         lats_random_sample,
                         c=clabels_random_sample,
                         marker='o',
                         cmap= plt.get_cmap('cividis'),
                         s=10.0,
                         transform=ccrs.Geodetic(),
                         )

        # fronts
        plt.plot(saf[:,0], saf[:,1], color="black", linewidth=2.0, transform=ccrs.Geodetic())
        plt.plot(pf[:,0], pf[:,1], color="blue", linewidth=2.0, transform=ccrs.Geodetic())
        plt.plot(saccf[:,0], saccf[:,1], color="green", linewidth=2.0, transform=ccrs.Geodetic())
        plt.plot(sbdy[:,0], sbdy[:,1], color="yellow", linewidth=2.0, transform=ccrs.Geodetic())

        #plt.colorbar(CS)
        ax.coastlines(resolution='50m')
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
        ax.add_feature(cartopy.feature.LAND)

        # save figure
        plt.savefig(dploc + 'i-metric_' + str(int(iclass)) + 'K.png', bbox_inches='tight')
        plt.savefig(dploc + 'i-metric_' + str(int(iclass)) + 'K.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

#############################################################################
# Wrapper for plotting dynamic height maps (each class, each pressure level)
#############################################################################
def plot_dynamic_height_maps(ploc, dfp, lon_range, lat_range, n_components_selected):

    # print
    print('plot_tools.plot_dynamic_height_maps')

    # p = 20 dbar
    dploc = ploc + 'dynamic_height/p0020dbar/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)
    # single pressure level
    dp1 = dfp.isel(depth=0)
    plot_hist_map(dploc, dp1, lon_range, lat_range, n_components_selected,
                  c_range=[dp1.dyn_height.min().values, dp1.dyn_height.max().values],
                  vartype='dyn_height')

    # p = 100 dbar
    dploc = ploc + 'dynamic_height/p0100dbar/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)
    # single pressure level
    dp1 = dfp.isel(depth=4)
    plot_hist_map(dploc, dp1, lon_range, lat_range, n_components_selected,
                  c_range=[dp1.dyn_height.min().values, dp1.dyn_height.max().values],
                  vartype='dyn_height')

    # p = 500 dbar
    dploc = ploc + 'dynamic_height/p0500dbar/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)
    # single pressure level
    dp1 = dfp.isel(depth=14)
    plot_hist_map(dploc, dp1, lon_range, lat_range, n_components_selected,
                  c_range=[dp1.dyn_height.min().values, dp1.dyn_height.max().values],
                   vartype='dyn_height')

    # p = 1000 dbar
    dploc = ploc + 'dynamic_height/p1000dbar/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)
    # single pressure level
    dp1 = dfp.isel(depth=20)
    plot_hist_map(dploc, dp1, lon_range, lat_range, n_components_selected,
                  c_range=[dp1.dyn_height.min().values, dp1.dyn_height.max().values],
                  vartype='dyn_height')

#####################################################################
# Plot a histogram map for each class
#####################################################################
def plot_hist_map(ploc, df1D,
                  lon_range, lat_range,
                  n_components_selected,
                  c_range=[0,1],
                  vartype='imetric',
                  colormap=plt.get_cmap('cividis'),
                  binsize=1,
                  bathy_fname='bathy.nc',
                  lev_range=range(-6000,1,500)):

    # print out
    print('plot_tools.plot_hist_map')

    # if plot sub directory doesn't exist, create it
    dploc = ploc + 'histogram_maps/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)

    # load bathymetry
    bds = io.load_bathymetry(bathy_fname)
    bathy_lon = bds['lon'][:]
    bathy_lat = bds['lat'][:]
    bathy = bds['bathy'][:]

    # load fronts
    pf = io.load_front("fronts/pf_kim.txt")
    saccf = io.load_front("fronts/saccf_kim.txt")
    saf = io.load_front("fronts/saf_kim.txt")
    sbdy = io.load_front("fronts/sbdy_kim.txt")

    # loop over classes, create one histogram plot per class
    for iclass in range(n_components_selected):

        # random sample for plotting
        df1 = df1D.where(df1D.label==iclass, drop=True)

        #colormap with Historical data
        plt.figure(figsize=(17, 13))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([lon_range[0], lon_range[1],
                       lat_range[0], lat_range[1]], ccrs.PlateCarree())

        # add bathymetry contours
        ax.contour(bathy_lon, bathy_lat, bathy, levels=lev_range,
                linewidths=0.5, alpha=0.5, colors="k", linestyles='-',
                transform=ccrs.PlateCarree())

        # define histogram, calculate mean i-metric value in each bin
        lon_bins = np.arange(lon_range[0], lon_range[1], binsize)
        lat_bins = np.arange(lat_range[0], lat_range[1], binsize)

        # select variable
        if vartype=="Tsurf":
            myVar = df1.prof_CT
        elif vartype=="Ssurf":
            myVar = df1.prof_SA
        elif vartype=="sig0surf":
            myVar = df1.sig0
        elif vartype=="Tmin":
            myVar = df1.Tmin
        elif vartype=="Smin":
            myVar = df1.Smin
        elif vartype=="sig0min":
            myVar = df1.sig0min
        elif vartype=="Tmax":
            myVar = df1.Tmax
        elif vartype=="Smax":
            myVar = df1.Smax
        elif vartype=="sig0max":
            myVar = df1.sig0max
        elif vartype=="imetric":
            myVar = df1.imetric
        elif vartype=="dyn_height":
            myVar = df1.dyn_height
        elif vartype=="mld":
            myVar = df1.mld
        else:
            print("Options include: Tsurf, Ssurf, sig0surf, Tmin, Smin, sig0min, \
                   Tmax, Smax, sig0max, imetric, dyn_height, mld")

        # histogram ()
        dA = (binsize*110e3)*(binsize*110e3*np.cos(df1.lat*np.pi/180))
        hist_denominator = histogram(df1.lon,
                                     df1.lat,
                                     bins=[lon_bins, lat_bins],
                                     weights=dA)
        hist_numerator = histogram(df1.lon,
                                   df1.lat,
                                   bins=[lon_bins, lat_bins],
                                   weights=myVar*dA)
        hiSsurf = hist_numerator/hist_denominator

        # colormesh histogram
        CS = plt.pcolormesh(lon_bins, lat_bins, hiSsurf.T,
                            transform=ccrs.PlateCarree(), cmap=colormap)
        plt.clim(c_range[0],c_range[1])

        # fronts
        h_saf = plt.plot(saf[:,0], saf[:,1], color="black", linewidth=2.0, transform=ccrs.Geodetic())
        h_pf = plt.plot(pf[:,0], pf[:,1], color="blue", linewidth=2.0, transform=ccrs.Geodetic())
        h_saccf = plt.plot(saccf[:,0], saccf[:,1], color="green", linewidth=2.0, transform=ccrs.Geodetic())
        h_sbdy = plt.plot(sbdy[:,0], sbdy[:,1], color="yellow", linewidth=2.0, transform=ccrs.Geodetic())

        # make two proxy artists to add to a legend
        l_saf = mpatches.Rectangle((0, 0), 1, 1, facecolor="black")
        l_pf = mpatches.Rectangle((0, 0), 1, 1, facecolor="blue")
        l_saccf = mpatches.Rectangle((0, 0), 1, 1, facecolor="green")
        l_sbdy = mpatches.Rectangle((0, 0), 1, 1, facecolor="yellow")
        labels = ['SAF', 'PF', 'SACCF', 'SBDY']
        plt.legend([l_saf, l_pf, l_saccf, l_sbdy], labels,
                   loc='lower right', fancybox=True)

        #plt.colorbar(CS)
        ax.coastlines(resolution='50m')
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
        ax.add_feature(cartopy.feature.LAND)

        # save figure
        plt.savefig(dploc + 'hist_' + vartype + '_' + str(int(iclass)) + 'K.png', bbox_inches='tight')
        plt.savefig(dploc + 'hist_' + vartype + '_' + str(int(iclass)) + 'K.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        # separate colorbar
        a = np.array([[c_range[0], c_range[1]]])
        plt.figure(figsize=(9, 1.5))
        img = plt.imshow(a, cmap=colormap)
        plt.gca().set_visible(False)
        cax = plt.axes([0.1, 0.2, 0.8, 0.6])
        cbar = plt.colorbar(orientation="horizontal", cax=cax)
        cbar.ax.tick_params(labelsize=22)
        plt.savefig(dploc + 'hist_' + vartype + 'colorbar.pdf', bbox_inches='tight')
        plt.savefig(dploc + 'hist_' + vartype + 'colorbar.png', bbox_inches='tight')
        plt.show()
        plt.close()

#####################################################################
# Plot sea ice freezing and fronts
#####################################################################
def plot_seaice_freezing(ploc=" ", lon_min=-65, lon_max=80, lat_min=-70, lat_max=-45):

    # print statement
    print('plot_tools.plot_seaice_freezing')

    # load winter sea ice freezing maps
    dp = io.load_sose_SIfreeze()

    # load fronts
    pf = io.load_front("fronts/pf_kim.txt")
    saccf = io.load_front("fronts/saccf_kim.txt")
    saf = io.load_front("fronts/saf_kim.txt")
    sbdy = io.load_front("fronts/sbdy_kim.txt")

    # make plot
    plt.figure(figsize=(17, 13))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    # sea ice freezing
    h = ax.contourf(dp.lon, dp.lat, dp.SIfreeze,
                    levels=[0.0, 2e-5, 4e-5, 6e-5, 8e-5, 10e-5],
                    transform=ccrs.PlateCarree(), vmin=0.0, vmax=10e-5,
                    cmap=cm.get_cmap("bone"))
    # fronts
    #plt.plot(pf[:,0], pf[:,1], color="blue", linewidth=2.0, transform=ccrs.Geodetic())
    plt.plot(saccf[:,0], saccf[:,1], color="green", linewidth=2.0, transform=ccrs.Geodetic())
    plt.plot(sbdy[:,0], sbdy[:,1], color="yellow", linewidth=2.0, transform=ccrs.Geodetic())
    ax.coastlines(resolution='50m')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    ax.add_feature(cartopy.feature.LAND)

    # save figure
    plt.savefig(ploc+"seaice/SeaIceFreezing_SOSE_winter.png", bbox_inches="tight")
    plt.savefig(ploc+"seaice/SeaIceFreezing_SOSE_winter.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

#####################################################################
# Plot t-SNE
#####################################################################
def plot_tsne(ploc, colormap, tSNE_data, colors_for_tSNE):

    print('plot_tools.plot_tsne')

    # scatterplot
    CS = plt.scatter(tSNE_data[0],
                     tSNE_data[1],
                     cmap=colormap,
                     s=10.0,
                     c=colors_for_tSNE)
    #plt.colorbar(CS)
    plt.title("t-SNE")
    plt.axis('tight')
    plt.savefig(ploc + 'tSNE' + '.png', bbox_inches='tight')
    plt.savefig(ploc + 'tSNE' + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# T-S plot for a single pressure level
#####################################################################
def plot_TS_single_lev(ploc, df, n_comp, colormap, descrip='', plev=0, PTrange=(-2, 27.0),
                       SPrange=(33.5, 37.5), lon = -20, lat = -65, rr = 0.60):

    print('plot_tools.plot_TS_single_lev')

    # subdirectory
    dploc = ploc + 'TSdiagrams/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)

    # import packages
    import gsw

    # make 1D
    df1D = df.isel(depth=plev)

    # define colormap
    #colormap = plt.get_cmap('Set1', n_comp)

    # grid
    pt_grid = np.linspace(PTrange[0],PTrange[1],100)
    sp_grid = np.linspace(SPrange[0],SPrange[1],100)
    p = df.depth.values[plev]
    lon = -20
    lat = -65

    sa_grid = gsw.SA_from_SP(sp_grid, p, lon, lat)
    ct_grid = gsw.CT_from_pt(sa_grid, pt_grid)
    ctg,sag = np.meshgrid(ct_grid,sa_grid)
    sig0_grid = gsw.density.sigma0(sag, ctg)

    # extract values as new DataArrays
    T = df1D.prof_CT.values
    S = df1D.prof_SA.values
    clabels = df1D.label.values

    # size of random sample (all profiles by now)
    random_sample_size = int(np.ceil(rr*df.profile.size))

    # random sample for plotting
    rows_id = random.sample(range(0,clabels.size-1), random_sample_size)
    T_random_sample = T[rows_id]
    S_random_sample = S[rows_id]
    clabels_random_sample = clabels[rows_id]

    #colormap with Historical data
    plt.figure(figsize=(13, 13))
    CL = plt.contour(sag, ctg, sig0_grid, colors='black', zorder=1)
    plt.clabel(CL, fontsize=24, inline=False, fmt='%.1f')
    SC = plt.scatter(S_random_sample,
                     T_random_sample,
                     c = clabels_random_sample,
                     marker='o',
                     cmap= colormap,
                     s=8.0,
                     zorder=2,
                     )
    plt.colorbar(SC)
    plt.ylabel('Conservative temperature [$^\circ$C]', fontsize=20)
    plt.xlabel('Absolute salinity [psu]', fontsize=20)
    plt.ylim(PTrange)
    plt.xlim(SPrange)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.title('T-S diagram at '+ str(p) + ' dbar', fontsize=22)
    plt.savefig(dploc + 'TS_single_lev_' + str(int(p)) + 'dbar' + descrip + '.png', bbox_inches='tight')
    plt.savefig(dploc + 'TS_single_lev_' + str(int(p)) + 'dbar' + descrip + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# T-S with the class means (and maybe stdevs) on them
#####################################################################
def plot_TS_withMeans(ploc, class_means, class_stds, n_comp, colormap, descrip='',
                      PTrange=(-2, 27.0), SPrange=(33.5, 37.5),
                      lon = -20, lat = -65):

    print('plot_tools.plot_TS_withMeans')

    # subdirectory
    dploc = ploc + 'TSdiagrams/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)

    # select colormap
    #colormap = plt.get_cmap('Set1', n_comp)
    cNorm = colors.Normalize(vmin=0, vmax=n_comp)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)

    # grid
    pt_grid = np.linspace(PTrange[0],PTrange[1],100)
    sp_grid = np.linspace(SPrange[0],SPrange[1],100)
    p = class_means.depth.values[0]
    lon = lon
    lat = lat
    sa_grid = gsw.SA_from_SP(sp_grid, p, lon, lat)
    ct_grid = gsw.CT_from_pt(sa_grid, pt_grid)
    ctg,sag = np.meshgrid(ct_grid,sa_grid)
    sig0_grid = gsw.density.sigma0(sag, ctg)

    # extract values as new DataArrays
    CTbar = class_means.prof_CT.values
    SAbar = class_means.prof_SA.values
    CTstd = class_stds.prof_CT.values
    SAstd = class_stds.prof_SA.values

    # TS diagram
    plt.figure(figsize=(13, 13))
    CL = plt.contour(sag, ctg, sig0_grid, colors='black', zorder=1)
    plt.clabel(CL, fontsize=24, inline=False, fmt='%.1f')
    for i in range(n_comp):
        colorVal = scalarMap.to_rgba(i)
        plt.plot(SAbar[i,:], CTbar[i,:],
                 linewidth=5.0, linestyle='solid', color=colorVal)
    plt.ylabel('Conservative temperature [$^\circ$C]', fontsize=20)
    plt.xlabel('Absolute salinity [psu]', fontsize=20)
    plt.ylim(PTrange)
    plt.xlim(SPrange)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    #plt.title('T-S diagram at '+ str(p) + ' dbar', fontsize=22)
    plt.savefig(dploc + 'TS_withMeans' + descrip + '.png', bbox_inches='tight')
    plt.savefig(dploc + 'TS_withMeans' + descrip + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# Single T-S plot featuring all pressure levels (shaded by class)
#####################################################################
def plot_TS_all_lev(ploc, df, n_comp, colormap, descrip='', PTrange=(-2, 27.0),
                    SPrange=(33.5, 37.5), lon = -20, lat = -65, rr = 0.33):

    print('plot_tools.plot_TS_all_lev')

    # subdirectory
    dploc = ploc + 'TSdiagrams/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)

    # make 1D by stacking and resetting index
    df1D = df.stack(z=('profile','depth')).reset_index('z')

    # define colormap
    #colormap = plt.get_cmap('Set1', n_comp)

    # grid
    pt_grid = np.linspace(PTrange[0],PTrange[1],100)
    sp_grid = np.linspace(SPrange[0],SPrange[1],100)
    p = df.depth.values[0]
    lon = -20
    lat = -65

    sa_grid = gsw.SA_from_SP(sp_grid, p, lon, lat)
    ct_grid = gsw.CT_from_pt(sa_grid, pt_grid)
    ctg,sag = np.meshgrid(ct_grid,sa_grid)
    sig0_grid = gsw.density.sigma0(sag, ctg)

    # extract values as new DataArrays
    T = df1D.prof_CT.values
    S = df1D.prof_SA.values
    clabels = df1D.label.values

    # size of random sample (all profiles by now)
    random_sample_size = int(np.ceil(rr*df1D.profile.size))

    # random sample for plotting
    rows_id = random.sample(range(0,clabels.size-1), random_sample_size)
    T_random_sample = T[rows_id]
    S_random_sample = S[rows_id]
    clabels_random_sample = clabels[rows_id]

    # T/S scatterplot where colors are class labels
    plt.figure(figsize=(13, 13))
    CL = plt.contour(sag, ctg, sig0_grid, colors='black', zorder=1)
    plt.clabel(CL, fontsize=24, inline=False, fmt='%.1f')
    SC = plt.scatter(S_random_sample,
                     T_random_sample,
                     c = clabels_random_sample,
                     marker='o',
                     cmap= colormap,
                     s=8.0,
                     zorder=2,
                     )
    plt.colorbar(SC)
    plt.ylabel('Conservative temperature [$^\circ$C]', fontsize=20)
    plt.xlabel('Absolute salinity [psu]', fontsize=20)
    plt.ylim(PTrange)
    plt.xlim(SPrange)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    #plt.title('T-S diagram at '+ str(p) + ' dbar', fontsize=22)
    plt.savefig(dploc + 'TS_all_levs' + descrip + '.png', bbox_inches='tight')
    plt.savefig(dploc + 'TS_all_levs' + descrip + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

#####################################################################
# T-S plot for a multiple pressure levels (one for each class)
#####################################################################
def plot_TS_multi_lev(ploc, df, n_comp, colormap, descrip='', plev=0, PTrange=(-2, 27.0),
                      SPrange=(33.5, 37.5), lon = -20, lat = -65, rr = 0.60):

    print('plot_tools.plot_TS_multi_lev')

    # subdirectory
    dploc = ploc + 'TSdiagrams/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)

    # make (stack and reset index)
    # stack command kills job when using "single class only" mode
    # dataset gets too big - maybe delete some things first?
    df1D = df.stack(z=('profile','depth')).reset_index('z')
    # now use isel to loop through labels

    # define colormap (fixed: 10 intervals in depth)
    #colormap = plt.get_cmap('cividis', 10)

    # grid for TS diagram
    pt_grid = np.linspace(PTrange[0],PTrange[1],100)
    sp_grid = np.linspace(SPrange[0],SPrange[1],100)
    p = df.depth.values[plev]
    lon = -20
    lat = -65

    # calculate SA and CT lines for plot
    sa_grid = gsw.SA_from_SP(sp_grid, p, lon, lat)
    ct_grid = gsw.CT_from_pt(sa_grid, pt_grid)
    ctg,sag = np.meshgrid(ct_grid,sa_grid)
    sig0_grid = gsw.density.sigma0(sag, ctg)

    # extract values as new DataArrays
    T = df1D.prof_CT.values
    S = df1D.prof_SA.values
    labels = df1D.label.values
    depths = df1D.depth.values

    # for each class, create new plot (shaded by depth)
    for nclass in range(n_comp):

        T1 = T[labels==nclass]
        S1 = S[labels==nclass]
        c1 = depths[labels==nclass] # shade by depth

        # size of random sample (all profiles by now)
        random_sample_size = int(np.ceil(rr*T1.size))

        # random sample for plotting
        rows_id = random.sample(range(0,T1.size-1), random_sample_size)
        T_random_sample = T1[rows_id]
        S_random_sample = S1[rows_id]
        clabels_random_sample = c1[rows_id]

        #colormap with Historical data
        plt.figure(figsize=(13, 13))
        CL = plt.contour(sag, ctg, sig0_grid, colors='black', zorder=1)
        plt.clabel(CL, fontsize=24, inline=False, fmt='%.1f')
        SC = plt.scatter(S_random_sample,
                         T_random_sample,
                         c = clabels_random_sample,
                         marker = 'o',
                         cmap = colormap,
                         s = 8.0,
                         zorder = 2,
                         )
        plt.colorbar(SC)
        plt.ylabel('Conservative temperature [$^\circ$C]', fontsize=20)
        plt.xlabel('Absolute salinity [psu]', fontsize=20)
        plt.ylim(PTrange)
        plt.xlim(SPrange)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        #plt.title('Class ' + str(nclass) , fontsize=22)
        plt.savefig(dploc + 'TS_multilev_class_' + str(nclass) + 'K' + descrip + '.png', bbox_inches='tight')
        plt.savefig(dploc + 'TS_multilev_class_' + str(nclass) + 'K' + descrip + '.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

#####################################################################
# T-S plot (shaded by time, one for each class)
#####################################################################
def plot_TS_bytime(ploc, df, n_comp, descrip='', plev=0, PTrange=(-2, 27.0),
                      SPrange=(33.5, 37.5), lon = -20, lat = -65, rr = 0.60,
                      timeShading='month'):

    print('plot_tools.plot_TS_bytime')

    # subdirectory
    dploc = ploc + 'TSdiagrams/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)

    # make (stack and reset index)
    df1D = df.isel(depth=0)
    # now use isel to loop through labels

    # define colormap (cyclic)
    if timeShading=='month':
        #colormap = plt.get_cmap('hsv_r', 12)
        colormap = cmocean.cm.phase
    else:
        colormap = plt.get_cmap('cividis', 30)

    # grid
    pt_grid = np.linspace(PTrange[0],PTrange[1],100)
    sp_grid = np.linspace(SPrange[0],SPrange[1],100)
    p = df.depth.values[plev]
    lon = -20
    lat = -65

    # calculate SA and CT lines for plot
    sa_grid = gsw.SA_from_SP(sp_grid, p, lon, lat)
    ct_grid = gsw.CT_from_pt(sa_grid, pt_grid)
    ctg,sag = np.meshgrid(ct_grid,sa_grid)
    sig0_grid = gsw.density.sigma0(sag, ctg)

    # time pre-processing
    time = pd.DatetimeIndex(df1D.time.values)

    # extract values as new DataArrays
    T = df1D.prof_CT.values
    S = df1D.prof_SA.values
    labels = df1D.label.values
    months = time.month.values
    years = time.year.values

    # for each class, create new plot
    for nclass in range(n_comp):

        T1 = T[labels==nclass]
        S1 = S[labels==nclass]

        if timeShading=='month':
            c1 = months[labels==nclass]
        elif timeShading=='year':
            c1 = years[labels==nclass]
        else:
            print('warning: shading must be year or month')

        # size of random sample (all profiles by now)
        random_sample_size = int(np.ceil(rr*T1.size))

        # random sample for plotting
        rows_id = random.sample(range(0,T1.size-1), random_sample_size)
        T_random_sample = T1[rows_id]
        S_random_sample = S1[rows_id]
        clabels_random_sample = c1[rows_id]

        #colormap with Historical data
        plt.figure(figsize=(13, 13))
        CL = plt.contour(sag, ctg, sig0_grid, colors='black', zorder=1)
        plt.clabel(CL, fontsize=24, inline=False, fmt='%.1f')
        SC = plt.scatter(S_random_sample,
                         T_random_sample,
                         c = clabels_random_sample,
                         marker='o',
                         cmap= colormap,
                         s=8.0,
                         zorder=2,
                         )
        plt.colorbar(SC)
        plt.ylabel('Conservative temperature [$^\circ$C]', fontsize=20)
        plt.xlabel('Absolute salinity [psu]', fontsize=20)
        plt.ylim(PTrange)
        plt.xlim(SPrange)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        #plt.title('Class ' + str(nclass) , fontsize=22)
        plt.savefig(dploc + 'TS_by' + timeShading + '_class_' + str(nclass) + 'K' + descrip + '.png', bbox_inches='tight')
        plt.savefig(dploc + 'TS_by' + timeShading + '_class_' + str(nclass) + 'K' + descrip + '.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

#####################################################################
# Volume histogram in T/S space
#####################################################################
def calc_and_plot_volume_histogram_TS(ploc, df,
                                      sbins = np.arange(31, 38, 0.025),
                                      tbins = np.arange(-2, 32, 0.1),
                                      modStr='', binsize=1):

    # print statement
    print('plot_tools.calc_and_plot_volume_histogram_TS')

    # subdirectory
    dploc = ploc + 'TSdiagrams/'
    if not os.path.exists(dploc):
        os.makedirs(dploc)

    # import packages
    import gsw

    # df1D (new)
    df1D = df.stack(z=('profile','depth')).reset_index('z')
    # drop unecessary variables for speed/efficiency
    df1D = df1D.drop({'time','year','month','CLASS','prof_date','prof_YYYYMMDD',
                      'prof_HHMMSS','sig0','label','posteriors'})

    # T-S grid for density reference lines
    ctg, sag = np.meshgrid(tbins , sbins)
    sig0_grid = gsw.density.sigma0(sag, ctg)

    # create histogram
    histTS = histogram(df1D.prof_SA, df1D.prof_CT, bins=[sbins,tbins])

    # histogram ()
    # --
    # --at present, this uses too much memory, causing a crash
    # --
    #dA = (binsize*110e3)*(binsize*110e3*np.cos(df.lat*np.pi/180))
    #hist_denominator = histogram(df1D.prof_SA,
    #                             df1D.prof_CT,
    #                             bins=[sbins, tbins],
    #                             weights=dA)
    #hist_numerator = histogram(df1D.prof_SA,
    #                           df1D.prof_CT,
    #                           bins=[sbins, tbins],
    #                           weights=df1D.depth*dA)
    #histTS_depth = hist_numerator/hist_denominator

    # scale the histogram (log10, transpose, scale by maximum)
    histTS_scaled = np.log10(histTS.T)
    histTS_scaled = histTS_scaled/histTS_scaled.max()

    # --- plot histogram
    plt.figure(figsize=(10,10))
    # now superimpose T-S contours
    TS = histTS_scaled.plot(levels=30)
    CL = plt.contour(sag, ctg, sig0_grid, colors='black', zorder=1)
    TS.colorbar.set_label('Count histogram (log10, scaled by maximum)')
    plt.clabel(CL, fontsize=14, inline=False, fmt='%.1f')
    plt.xlabel('Absolute salinity (psu)')
    plt.ylabel('Conservative temperature (°C)')
    plt.savefig(dploc + 'histogram_depth' + modStr + '.png', bbox_inches='tight')
    plt.savefig(dploc + 'histogram_depth' + modStr + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    return histTS_scaled

#####################################################################
# Plot class stats, split by longitude
#####################################################################
def plot_lon_split(ploc, df):

    print('plot_tools.plot_lon_split')

    fs = 28 # font size

    dfg = df.groupby('subgroup')
    z = df.depth.values
    sig0_levs = df.sig0_levs.values

    dfg_mean = dfg.mean('profile')
    dfg_std = dfg.std('profile')

    ### CT
    myVar = 'CT'
    ybar = dfg_mean[myVar]
    ystd = dfg_std[myVar]
    # plot
    plt.figure(figsize=(25,30))
    ax = plt.subplot(1,2,1)
    # group a
    myGroup = 'a'
    plt.plot(ybar.sel(subgroup=myGroup), -z,
                marker='', linestyle='solid', color='b', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)+ystd.sel(subgroup=myGroup), -z,
                marker='', linestyle='dashed', color='b', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)-ystd.sel(subgroup=myGroup), -z,
                marker='', linestyle='dashed', color='b', linewidth=6.0, alpha=0.9)
    plt.grid()
    # group b
    myGroup = 'b'
    plt.plot(ybar.sel(subgroup=myGroup), -z,
                marker='', linestyle='solid', color='r', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)+ystd.sel(subgroup=myGroup), -z,
                marker='', linestyle='dashed', color='r', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)-ystd.sel(subgroup=myGroup), -z,
                marker='', linestyle='dashed', color='r', linewidth=6.0, alpha=0.9)
    plt.title('Blue: <= lon0, Red: > lon0', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    # right panel
    ax = plt.subplot(1,2,2)
    dybar = ybar.sel(subgroup='b') - ybar.sel(subgroup='a')
    plt.plot(dybar, -z,
                marker='', linestyle='solid', color='black', linewidth=6.0, alpha=0.9)
    plt.title('Red-Blue', fontsize=fs)
    plt.grid()
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.savefig(ploc + 'twogroup_CT.png', bbox_inches='tight')
    plt.savefig(ploc + 'twogroup_CT.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    ### SA
    myVar = 'SA'
    ybar = dfg_mean[myVar]
    ystd = dfg_std[myVar]
    # plot
    plt.figure(figsize=(25,30))
    ax = plt.subplot(1,2,1)
    # group a
    myGroup = 'a'
    plt.plot(ybar.sel(subgroup=myGroup), -z,
                marker='', linestyle='solid', color='b', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)+ystd.sel(subgroup=myGroup), -z,
                marker='', linestyle='dashed', color='b', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)-ystd.sel(subgroup=myGroup), -z,
                marker='', linestyle='dashed', color='b', linewidth=6.0, alpha=0.9)
    plt.grid()
    # group b
    myGroup = 'b'
    plt.plot(ybar.sel(subgroup=myGroup), -z,
                marker='', linestyle='solid', color='r', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)+ystd.sel(subgroup=myGroup), -z,
                marker='', linestyle='dashed', color='r', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)-ystd.sel(subgroup=myGroup), -z,
                marker='', linestyle='dashed', color='r', linewidth=6.0, alpha=0.9)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.title('Blue: <= lon0, Red: > lon0', fontsize=fs)

    # right panel
    ax = plt.subplot(1,2,2)
    dybar = ybar.sel(subgroup='b') - ybar.sel(subgroup='a')
    plt.plot(dybar, -z,
                marker='', linestyle='solid', color='black', linewidth=6.0, alpha=0.9)
    plt.title('Red-Blue', fontsize=fs)
    plt.grid()
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.savefig(ploc + 'twogroup_SA.png', bbox_inches='tight')
    plt.show()
    plt.close()

    ### sig0
    myVar = 'sig0'
    ybar = dfg_mean[myVar]
    ystd = dfg_std[myVar]
    # plot
    plt.figure(figsize=(25,30))
    ax = plt.subplot(1,2,1)
    # group a
    myGroup = 'a'
    plt.plot(ybar.sel(subgroup=myGroup), -z,
                marker='', linestyle='solid', color='b', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)+ystd.sel(subgroup=myGroup), -z,
                marker='', linestyle='dashed', color='b', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)-ystd.sel(subgroup=myGroup), -z,
                marker='', linestyle='dashed', color='b', linewidth=6.0, alpha=0.9)
    plt.grid()
    # group b
    myGroup = 'b'
    plt.plot(ybar.sel(subgroup=myGroup), -z,
                marker='', linestyle='solid', color='r', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)+ystd.sel(subgroup=myGroup), -z,
                marker='', linestyle='dashed', color='r', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)-ystd.sel(subgroup=myGroup), -z,
                marker='', linestyle='dashed', color='r', linewidth=6.0, alpha=0.9)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.title('Blue: <= lon0, Red: > lon0', fontsize=fs)

    # right panel
    ax = plt.subplot(1,2,2)
    dybar = ybar.sel(subgroup='b') - ybar.sel(subgroup='a')
    plt.plot(dybar, -z,
                marker='', linestyle='solid', color='black', linewidth=6.0, alpha=0.9)
    plt.title('Red-Blue', fontsize=fs)
    plt.grid()
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.savefig(ploc + 'twogroup_sig0.png', bbox_inches='tight')
    plt.show()
    plt.close()

    ### CT_onsig
    myVar = 'CT_onsig'
    ybar = dfg_mean[myVar]
    ystd = dfg_std[myVar]
    # plot
    plt.figure(figsize=(25,30))
    ax = plt.subplot(1,2,1)
    # group a
    myGroup = 'a'
    plt.plot(ybar.sel(subgroup=myGroup), sig0_levs,
                marker='', linestyle='solid', color='b', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)+ystd.sel(subgroup=myGroup), sig0_levs,
                marker='', linestyle='dashed', color='b', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)-ystd.sel(subgroup=myGroup), sig0_levs,
                marker='', linestyle='dashed', color='b', linewidth=6.0, alpha=0.9)
    plt.grid()
    # group b
    myGroup = 'b'
    plt.plot(ybar.sel(subgroup=myGroup), sig0_levs,
                marker='', linestyle='solid', color='r', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)+ystd.sel(subgroup=myGroup), sig0_levs,
                marker='', linestyle='dashed', color='r', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)-ystd.sel(subgroup=myGroup), sig0_levs,
                marker='', linestyle='dashed', color='r', linewidth=6.0, alpha=0.9)

    plt.title('Blue: <= lon0, Red: > lon0', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    # right panel
    ax = plt.subplot(1,2,2)
    dybar = ybar.sel(subgroup='b') - ybar.sel(subgroup='a')
    plt.plot(dybar, sig0_levs,
                marker='', linestyle='solid', color='black', linewidth=6.0, alpha=0.9)
    plt.title('Red-Blue', fontsize=fs)
    plt.grid()
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.savefig(ploc + 'twogroup_CT_onSig0.png', bbox_inches='tight')
    plt.show()
    plt.close()

    ### SA_onsig
    myVar = 'SA_onsig'
    ybar = dfg_mean[myVar]
    ystd = dfg_std[myVar]
    # plot
    plt.figure(figsize=(25,30))
    ax = plt.subplot(1,2,1)
    # group a
    myGroup = 'a'
    plt.plot(ybar.sel(subgroup=myGroup), sig0_levs,
                marker='', linestyle='solid', color='b', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)+ystd.sel(subgroup=myGroup), sig0_levs,
                marker='', linestyle='dashed', color='b', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)-ystd.sel(subgroup=myGroup), sig0_levs,
                marker='', linestyle='dashed', color='b', linewidth=6.0, alpha=0.9)
    plt.grid()
    # group b
    myGroup = 'b'
    plt.plot(ybar.sel(subgroup=myGroup), sig0_levs,
                marker='', linestyle='solid', color='r', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)+ystd.sel(subgroup=myGroup), sig0_levs,
                marker='', linestyle='dashed', color='r', linewidth=6.0, alpha=0.9)
    plt.plot(ybar.sel(subgroup=myGroup)-ystd.sel(subgroup=myGroup), sig0_levs,
                marker='', linestyle='dashed', color='r', linewidth=6.0, alpha=0.9)

    plt.title('Blue: <= lon0, Red: > lon0', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    # right panel
    ax = plt.subplot(1,2,2)
    dybar = ybar.sel(subgroup='b') - ybar.sel(subgroup='a')
    plt.plot(dybar, sig0_levs,
                marker='', linestyle='solid', color='black', linewidth=6.0, alpha=0.9)
    plt.title('Red-Blue', fontsize=fs)
    plt.grid()
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.savefig(ploc + 'twogroup_SA_onSig0.png', bbox_inches='tight')
    plt.show()
    plt.close()
