#####################################################################
# Utilities for loading profile data, slicing
#####################################################################

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import random
import gsw

def plot_TS_single_lev(df, PTrange=(-2, 2.1), SPrange=(33.8, 34.8), 
                       lon = -20, lat = -65, rr = 0.60, colormap):
    
    # grid
    pt_grid = np.linspace(PTrange[0],PTrange[1],100)
    sp_grid = np.linspace(SPrange[0],SPrange[1],100)
    p = df.depth.values
    lon = -20 
    lat = -65
    
    sa_grid = gsw.SA_from_SP(sp_grid, p, lon, lat)
    ct_grid = gsw.CT_from_pt(sa_grid, pt_grid)
    ctg,sag = np.meshgrid(ct_grid,sa_grid)
    sig0_grid = gsw.density.sigma0(sag, ctg)
    
    # extract values as new DataArrays
    T = df.prof_CT.values
    S = df.prof_SA.values
    clabels = df.label.values
    
    # size of random sample (all profiles by now)
    random_sample_size = int(np.ceil(rr*df.profile.size))
    
    # random sample for plotting
    rows_id = random.sample(range(0,clabels.size-1), random_sample_size)
    T_random_sample = T[rows_id]
    S_random_sample = S[rows_id]
    clabels_random_sample = clabels[rows_id]
    
    #colormap with Historical data
    plt.figure(figsize=(13, 13))
    CL = plt.contour(ctg, sag, sig0_grid, colors='black', zorder=1)
    plt.clabel(CL, fontsize=24, inline=False, fmt='%.1f')
    plt.scatter(T_random_sample, 
                     S_random_sample, 
                     c = clabels_random_sample,
                     marker='o',
                     cmap= colormap,
                     s=8.0,
                     zorder=2,
                     )
    
    plt.xlabel('Conservative temperature [$^\circ$C]', fontsize=20)
    plt.ylabel('Absolute salinity [psu]', fontsize=20)
    plt.xlim(PTrange)
    plt.ylim(SPrange)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('T-S diagram at '+ str(p) + ' dbar', fontsize=22)




