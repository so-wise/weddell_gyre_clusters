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
import pandas as pd
from sklearn import manifold
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.ops import cascaded_union
import random
import gsw

#####################################################################
# Plot single profile
#####################################################################
def plot_profile(ploc, df):
   fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=[8,5])
   df.prof_T.plot(ax=ax1, y='depth', yincrease=False)
   df.prof_S.plot(ax=ax2, y='depth', yincrease=False)
   df.swap_dims({'depth': 'sig0'}).prof_CT.plot(ax=ax3, y='sig0', marker='.', yincrease=False)
   df.swap_dims({'depth': 'sig0'}).prof_SA.plot(ax=ax4, y='sig0', marker='.', yincrease=False)
   #fig.subplots_adjust(wspace=0.7)
   # save figure and close
   plt.savefig(ploc + 'single_profile.png', bbox_inches='tight')
   plt.close()

#####################################################################
# Plot single profile
#####################################################################
def plot_profiles_on_density_levels(ploc, profiles, frac=0.33):

    # start message
    print('plot_tools.plot_profiles_on_density_levels')

    # conservative temp
    plt.figure(figsize=(30, 30))
    profiles.ct_on_sig0.plot()
    plt.savefig(ploc + 'ct_on_sig0.png', bbox_inches='tight')
    plt.close()

    # absolute salinity
    plt.figure(figsize=(30, 30))
    profiles.sa_on_sig0.plot()
    plt.savefig(ploc + 'sa_on_sig0.png', bbox_inches='tight')
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
    plt.close()

#####################################################################
# Plot PCA structure in 3D space (with or without class labels
#####################################################################
def plot_pca3D(ploc, colormap, profiles, Xpca, frac=0.33, withLabels=False):

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

    # 3D scatterplots

    # view 1
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, 0)
    CS = ax.scatter(xyp[:,0], xyp[:,1], xyp[:,2], 'o', c=c, cmap=colormap, alpha=0.1, s=1.0)
    if withLabels==True: plt.colorbar(CS)
    plt.savefig(ploc + 'pca_scatter_' + labs + '_view1' + '.png', bbox_inches='tight')
    plt.close()

    # view 2
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, 120)
    CS = ax.scatter(xyp[:,0], xyp[:,1], xyp[:,2], 'o', c=c, cmap=colormap, alpha=0.1, s=1.0)
    if withLabels==True: plt.colorbar(CS)
    plt.savefig(ploc + 'pca_scatter_' + labs + '_view2' + '.png', bbox_inches='tight')
    plt.close()

    # view 3
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, 240)
    CS = ax.scatter(xyp[:,0], xyp[:,1], xyp[:,2], 'o', c=c, cmap=colormap, alpha=0.1, s=1.0)
    if withLabels==True: plt.colorbar(CS)
    plt.savefig(ploc + 'pca_scatter_' + labs + '_view3' + '.png', bbox_inches='tight')
    plt.close()

#####################################################################
# Pairplot (general)
#####################################################################
def plot_pairs(ploc, dataset, kind="hist", descr=""):

    # create pandas dataframe from numpy array
    df = pd.DataFrame(data=dataset)

    # create figure
    fig = plt.figure(figsize=(15,15))
    sns.pairplot(df, kind=kind)
    plt.savefig(ploc + 'pairplot_' + descr + '.png', bbox_inches='tight')
    plt.close()

#####################################################################
# Plot UMAP structures (not shaded by class or anything yet)
#####################################################################
def plot_umap(ploc, Xtrans, frac=0.33):

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
    plt.close()

    # view 2
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, 120)
    ax.scatter(xyp[:,0], xyp[:,1], xyp[:,2], 'o', s=1.0)
    plt.savefig(ploc + 'umap_scatter_nolabels_view2' + '.png', bbox_inches='tight')
    plt.close()

    # view 3
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, 240)
    ax.scatter(xyp[:,0], xyp[:,1], xyp[:,2], 'o', s=1.0)
    plt.savefig(ploc + 'umap_scatter_nolabels_view3' + '.png', bbox_inches='tight')
    plt.close()

#####################################################################
# Plot BIC scores
#####################################################################
def plot_bic_scores(ploc, max_N, bic_mean, bic_std):
    # plot the BIC scores
    n_components_range = range(2, max_N)
    plt.figure(figsize=(20, 8))
    spl = plt.subplot(2, 1, 1)
    plt.plot(n_components_range, bic_mean-bic_std, '--')
    plt.plot(n_components_range, bic_mean, '-')
    plt.plot(n_components_range, bic_mean+bic_std, '--')
    plt.xticks(n_components_range)
    plt.title('BIC score per model', fontsize=18)
    spl.set_xlabel('Number of components',fontsize=18)
    spl.set_ylabel('BIC score',fontsize=18)
    # save figure
    plt.savefig(ploc + 'bic_scores.png', bbox_inches='tight')
    plt.close()

#####################################################################
# Plot AIC scores
#####################################################################
def plot_aic_scores(ploc, max_N, aic_mean, aic_std):
    # plot the AIC scores
    n_components_range = range(2, max_N)
    plt.figure(figsize=(20, 8))
    spl = plt.subplot(2, 1, 1)
    plt.plot(n_components_range, aic_mean-aic_std, '--')
    plt.plot(n_components_range, aic_mean, '-')
    plt.plot(n_components_range, aic_mean+aic_std, '--')
    plt.xticks(n_components_range)
    plt.title('AIC score per model', fontsize=18)
    spl.set_xlabel('Number of components',fontsize=18)
    spl.set_ylabel('AIC score',fontsize=18)
    # save figure
    plt.savefig(ploc + 'aic_scores.png', bbox_inches='tight')
    plt.close()

#####################################################################
# Sample profile plots for T, S, CT, SA, and sig0
#####################################################################
def prof_TS_sample_plots(ploc, profiles):

    # plotting subset (just a few for visualisation purposes)
    subset = range(1000,2000,1)
    # temperature plots
    fig, ax = plt.subplots(figsize=(15,10))
    profiles.prof_T[subset].plot(y='depth', yincrease=False)
    plt.savefig(ploc + 'prof_T_subset.png', bbox_inches='tight')
    plt.close()
    # salinity plots
    fig, ax = plt.subplots(figsize=(15,10))
    profiles.prof_S[subset].plot(y='depth', yincrease=False)
    plt.savefig(ploc + 'prof_S_subset.png', bbox_inches='tight')
    plt.close()
    # conservative temperature plots
    fig, ax = plt.subplots(figsize=(15,10))
    profiles.prof_CT[subset].plot(y='depth', yincrease=False)
    plt.savefig(ploc + 'prof_CT_subset.png', bbox_inches='tight')
    plt.close()
    # absolute salinity plots
    fig, ax = plt.subplots(figsize=(15,10))
    profiles.prof_SA[subset].plot(y='depth', yincrease=False)
    plt.savefig(ploc + 'prof_SA_subset.png', bbox_inches='tight')
    plt.close()
    # sigma0 density
    fig, ax = plt.subplots(figsize=(15,10))
    profiles.sig0[subset].plot(y='depth', yincrease=False)
    plt.savefig(ploc + 'sig0_subset.png', bbox_inches='tight')
    plt.close()
    # histograms
    xr.plot.hist(profiles.prof_T,figsize=(15,10))
    plt.savefig(ploc + 'hist_prof_T.png', bbox_inches='tight')
    plt.close()
    xr.plot.hist(profiles.prof_S,figsize=(15,10))
    plt.savefig(ploc + 'hist_prof_S.png', bbox_inches='tight')
    plt.close()
    xr.plot.hist(profiles.prof_CT,figsize=(15,10))
    plt.savefig(ploc + 'hist_prof_CT.png', bbox_inches='tight')
    plt.close()
    xr.plot.hist(profiles.prof_SA,figsize=(15,10))
    plt.savefig(ploc + 'hist_prof_SA.png', bbox_inches='tight')
    plt.close()
    xr.plot.hist(profiles.sig0,figsize=(15,10))
    plt.savefig(ploc + 'hist_sig0.png', bbox_inches='tight')
    plt.close()

#####################################################################
# Plot PCA structure for temperature and salinity
#####################################################################
def plot_pca_structure(ploc, profiles, pca, number_of_pca_components, zmin, zmax):

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
    plt.close()

#####################################################################
# Plot mean and stdev salinity class structure
#####################################################################
def plot_SA_class_structure(ploc, profiles, class_means,
                           class_stds, n_components_selected,
                           zmin, zmax, Smin=33.6, Smax=37.0):

    # select colormap
    colormap = plt.get_cmap('tab20', n_components_selected)
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
        plt.title('Class = ' + str(num), fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    #fig.subplots_adjust(wspace=0.7)
    plt.savefig(ploc + 'prof_SA_byClass.png', bbox_inches='tight')
    plt.close()

#####################################################################
# Plot mean and stdev conservative temperature class structure
#####################################################################
def plot_CT_class_structure(ploc, profiles, class_means,
                            class_stds, n_components_selected,
                            zmin, zmax, Tmin=-3, Tmax=20):

    # select colormap
    colormap = plt.get_cmap('tab20', n_components_selected)
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
        plt.title('Class = ' + str(num), fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    #fig.subplots_adjust(wspace=0.7)
    plt.savefig(ploc + 'prof_CT_byClass.png', bbox_inches='tight')
    plt.close()

#####################################################################
# Plot mean and stdev salinity class structure (on sigma)
#####################################################################
def plot_SA_class_structure_onSig(ploc, profiles, class_means,
                           class_stds, n_components_selected,
                           Smin=33.6, Smax=37.0):

    # select colormap
    colormap = plt.get_cmap('tab20', n_components_selected)
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
        plt.ylabel('$\sigma_0$ (kg/m$^3$)', fontsize=fs)
        plt.title('Class = ' + str(num), fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    #fig.subplots_adjust(wspace=0.7)
    plt.savefig(ploc + 'prof_SA_sig0_byClass.png', bbox_inches='tight')
    plt.close()

#####################################################################
# Plot mean and stdev conservative temperature class structure
#####################################################################
def plot_CT_class_structure_onSig(ploc, profiles, class_means,
                            class_stds, n_components_selected,
                            Tmin=-3, Tmax=20):

    # select colormap
    colormap = plt.get_cmap('tab20', n_components_selected)
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
        plt.ylabel('$\sigma_0$ (kg/m$^3$)', fontsize=fs)
        plt.title('Class = ' + str(num), fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    #fig.subplots_adjust(wspace=0.7)
    plt.savefig(ploc + 'prof_CT_sig0_byClass.png', bbox_inches='tight')
    plt.close()

#####################################################################
# Plot mean and stdev density class structure
#####################################################################
def plot_sig0_class_structure(ploc, profiles, class_means,
                           class_stds, n_components_selected,
                           zmin, zmax, sig0min=24.0, sig0max=28.0):

    # select colormap
    colormap = plt.get_cmap('tab20', n_components_selected)
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
        plt.title('Class = ' + str(num), fontsize=fs)

        # font and axis stuff
        plt.gca().invert_yaxis()
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

    #fig.subplots_adjust(wspace=0.7)
    plt.savefig(ploc + 'prof_sig0_byClass.png', bbox_inches='tight')
    plt.close()

#####################################################################
# Plot class label map using cartopy
#####################################################################
def plot_label_map(ploc, profiles, n_components_selected,
                   lon_min, lon_max, lat_min, lat_max):

    # define colormap
    colormap = plt.get_cmap('tab20', n_components_selected)

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

    # add background image for ocean bathymetry
    ax.stock_img()

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
    ax.gridlines(color='black')
    ax.add_feature(cartopy.feature.LAND)
    #plt.colorbar(CS)

    # 200 m bathymetry line
    #bathym = cfeature.NaturalEarthFeature(name='bathymetry_J_200', scale='10m', category='physical')
    #bathym = cascaded_union(list(bathym.geometries()))
    #ax.add_geometries(bathym, facecolor='none', edgecolor='grey', linestyle='solid', linewidth=0.5, crs=ccrs.PlateCarree())

    # save figure
    plt.savefig(ploc + 'label_map.png', bbox_inches='tight')
    plt.close()

#####################################################################
# Plot single i-metric map
#####################################################################
def plot_i_metric_single_panel(ploc, df1D, lon_min, lon_max, lat_min, lat_max, rr=0.66):

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
    CS = plt.scatter(lons_random_sample-360,
                     lats_random_sample,
                     c=clabels_random_sample,
                     marker='o',
                     cmap= plt.get_cmap('cividis'),
                     s=10.0,
                     transform=ccrs.Geodetic(),
                     )
    ax.coastlines(resolution='50m')
    ax.gridlines(color='black')
    ax.add_feature(cartopy.feature.LAND)

    # save figure
    plt.savefig(ploc + 'i-metric_single.png', bbox_inches='tight')
    plt.close()

#####################################################################
# Plot multiple i-metric maps (one per class)
#####################################################################
def plot_i_metric_multiple_panels(ploc, df1D, lon_min, lon_max,
                                    lat_min, lat_max, n_components_selected):

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
        CS = plt.scatter(lons_random_sample-360,
                         lats_random_sample,
                         c=clabels_random_sample,
                         marker='o',
                         cmap= plt.get_cmap('cividis'),
                         s=10.0,
                         transform=ccrs.Geodetic(),
                         )
        #plt.colorbar(CS)
        ax.coastlines(resolution='50m')
        ax.gridlines(color='black')
        ax.add_feature(cartopy.feature.LAND)

        # save figure
        plt.savefig(ploc + 'i-metric_' + str(int(iclass)) + 'K.png', bbox_inches='tight')
        plt.close()

#####################################################################
# Plot t-SNE
#####################################################################
def plot_tsne(ploc, colormap, tSNE_data, colors_for_tSNE):

    # scatterplot
    CS = plt.scatter(tSNE_data[0],
                     tSNE_data[1],
                     cmap=colormap,
                     s=5.0,
                     c=colors_for_tSNE)
    plt.colorbar(CS)
    plt.title("t-SNE")
    plt.axis('tight')
    plt.savefig(ploc + 'tSNE' + '.png', bbox_inches='tight')
    plt.close()

#####################################################################
# T-S plot for a single pressure level
#####################################################################
def plot_TS_single_lev(ploc, df, n_comp, descrip='', plev=0, PTrange=(-2, 27.0),
                       SPrange=(33.5, 37.5), lon = -20, lat = -65, rr = 0.60):

    # import packages
    import gsw

    # make 1D
    df1D = df.isel(depth=plev)

    # define colormap
    colormap = plt.get_cmap('tab20', n_comp)

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
    plt.savefig(ploc + 'TS_single_lev_' + str(int(p)) + 'dbar' + descrip + '.png', bbox_inches='tight')
    plt.close()

#####################################################################
# T-S plot for all pressure levels
#####################################################################
def plot_TS_all_lev(ploc, df, n_comp, descrip='', PTrange=(-2, 27.0),
                    SPrange=(33.5, 37.5), lon = -20, lat = -65, rr = 0.33):

    # make 1D by stacking and resetting index
    df1D = df.stack(z=('profile','depth')).reset_index('z')

    # define colormap
    colormap = plt.get_cmap('tab20', n_comp)

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
    #plt.title('T-S diagram at '+ str(p) + ' dbar', fontsize=22)
    plt.savefig(ploc + 'TS_all_levs' + descrip + '.png', bbox_inches='tight')
    plt.close()

#####################################################################
# T-S plot for a multiple pressure levels (one for each class)
#####################################################################
def plot_TS_multi_lev(ploc, df, n_comp, descrip='', plev=0, PTrange=(-2, 27.0),
                      SPrange=(33.5, 37.5), lon = -20, lat = -65, rr = 0.60):

    # make (stack and reset index)
    df1D = df.stack(z=('profile','depth')).reset_index('z')
    # now use isel to loop through labels

    # define colormap
    colormap = plt.get_cmap('cividis', n_comp)

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

    # extract values as new DataArrays
    T = df1D.prof_CT.values
    S = df1D.prof_SA.values
    labels = df1D.label.values
    depths = df1D.depth.values

    # for each class, create new plot
    for nclass in range(n_comp):

        T1 = T[labels==nclass]
        S1 = S[labels==nclass]
        c1 = depths[labels==nclass]

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
        plt.title('Class ' + str(nclass) , fontsize=22)
        plt.savefig(ploc + 'TS_multilev_class_' + str(nclass) + 'K' + descrip + '.png', bbox_inches='tight')
        plt.close()

#####################################################################
# T-S plot (shaded by time, one for each class)
#####################################################################
def plot_TS_bytime(ploc, df, n_comp, descrip='', plev=0, PTrange=(-2, 27.0),
                      SPrange=(33.5, 37.5), lon = -20, lat = -65, rr = 0.60,
                      timeShading='month'):

    # make (stack and reset index)
    df1D = df.isel(depth=0)
    # now use isel to loop through labels

    # define colormap (cyclic)
    if timeShading=='month':
        colormap = plt.get_cmap('hsv_r', 12)
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
        plt.title('Class ' + str(nclass) , fontsize=22)
        plt.savefig(ploc + 'TS_by' + timeShading + '_class_' + str(nclass) + 'K' + descrip + '.png', bbox_inches='tight')
        plt.close()

#####################################################################
# Plot class stats, split by longitude
#####################################################################
def plot_lon_split(ploc, df):

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
    plt.close()
