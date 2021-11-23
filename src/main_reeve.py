#
# --- manually for now; need to get into our own Docker image
# pip install umap-learn cmocean gsw seaborn
#

# local code
import load_and_preprocess as lp
import bic_and_aic as bic
import plot_tools as pt
import gmm
# plotting tools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
# scikit-learn
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn import manifold
from sklearn import mixture
# xarray, numpy, other tools
import xarray as xr
import numpy as np
import umap

# load Reeve climatology
ds = lp.load_reeve_climatology()

# select subset (by pressure)
ds = ds.sel(level=slice(300,1000))

# plot mean
ds_mean = ds.mean(dim="profile", skipna=True)
ds_mean.prof_CT.plot(y="level", yincrease=False)
plt.savefig('plots_reeve/meanCT.png',bbox_inches='tight')
ds_mean.prof_SA.plot(y="level", yincrease=False)
plt.savefig('plots_reeve/meanSA.png',bbox_inches='tight')

# convert to numpy
pvals = ds.level.values
CT = ds.prof_CT.values
SA = ds.prof_SA.values

# figure of all profiles
fig, (ax1, ax2) = plt.subplots(1, 2)
for j in range(CT.shape[1]):
    CT1 = CT[:,j]
    ax1.plot(CT1,pvals,color='grey',alpha=0.1)
    SA1 = SA[:,j]
    ax2.plot(SA1,pvals,color='grey',alpha=0.1)
plt.savefig('plots_reeve/allProfs.png',bbox_inches='tight')

#
# Training the GMM
#

# try to get rid of NaN values
CTnn = CT[:, ~np.isnan(CT).any(axis=0)].transpose()
SAnn = SA[:, ~np.isnan(SA).any(axis=0)].transpose()

# scale salinity and temperature
scaled_CT = preprocessing.scale(CTnn)
scaled_SA = preprocessing.scale(SAnn)

# concatenate
Xscaled = np.concatenate((scaled_CT,scaled_SA),axis=1)

# apply pca
pca = PCA(n_components=6)
pca.fit(Xscaled)
Xpca = pca.transform(Xscaled)
total_variance_explained_ = np.sum(pca.explained_variance_ratio_)
print(total_variance_explained_)

#bic_mean, bic_std, aic_mean, aic_std = bic.calc_bic_and_aic(Xpca, 20, max_iter=20)
#pt.plot_bic_scores('plots_reeve/', 20, bic_mean, bic_std)
#pt.plot_aic_scores('plots_reeve/', 20, aic_mean, aic_std)

# gmm
n_components_selected = 7
myGMM = gmm.train_gmm(Xpca, n_components_selected = n_components_selected)

#
# --- Labelling step (_ls)
#

# The below seems to work. Next step is to get it associated with the xarray DataArrays above

scaled_CT_ls = preprocessing.scale(CT.transpose())
scaled_SA_ls = preprocessing.scale(SA.transpose())
Xscaled_ls = np.concatenate((scaled_CT_ls,scaled_SA_ls),axis=1)

labels_ls = np.empty((Xscaled_ls.shape[0],))
post_primus = np.empty((Xscaled_ls.shape[0],))
post_secundo = np.empty((Xscaled_ls.shape[0],))
iMetric = np.empty((Xscaled_ls.shape[0],))

j = -1
for column in Xscaled_ls:
    j = j + 1
    if np.isnan(column).any()==True:
        myLabel = np.nan
        labels_ls[j] = myLabel
    else:
        myPCA = pca.transform(column.reshape(1,-1))
        myLabel = myGMM.predict(myPCA)
        myPost = myGMM.predict_proba(myPCA)
        # assign label
        labels_ls[j] = myLabel
        # posterior probabilities
        myPost.sort()
        post_primus[j] = myPost[0][-1]
        post_secundo[j] = myPost[0][-2]
        iMetric[j] = 1 - (post_primus[j] - post_secundo[j])

# sanitise
eps = 0.0001
post_primus[post_primus<eps] = 0.0
post_secundo[post_secundo<eps] = 0.0
iMetric[iMetric<eps] = 0.0
# nans
post_primus[np.isnan(labels_ls)] = np.nan
post_secundo[np.isnan(labels_ls)] = np.nan
iMetric[np.isnan(labels_ls)] = np.nan

# add labels and max posterior prob. DataArrays to the Dataset
ds['label'] = xr.DataArray(labels_ls, dims=['profile'])

### EVERYTHING WORKS FINE UNTIL I UNCOMMENT THE LINES below
### FOR SOME REASON, THEY CLOBBER THE MULTIINDEX AND TURN IT INTO AN object
#ds['post_primus'] = xr.DataArray(post_primus, dims=['profile'])
#ds['post_secundo'] = xr.DataArray(post_secundo, dims=['profile'])
#ds['iMetric'] = xr.DataArray(iMetric, dims=['profile'])

# finally, unstack!
ds = ds.unstack()

#
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

c = ax0.pcolor(ds.label[0,:,:].values)
ax0.set_title('2001-2005')

c = ax1.pcolor(ds.label[1,:,:].values)
ax1.set_title('2006-2009')

c = ax2.pcolor(ds.label[2,:,:].values)
ax2.set_title('2010-2013')

fig.tight_layout()
plt.show()
plt.savefig('plots_reeve/map_testing.png', bbox_inches='tight')

##
fig0, ax0 = plt.subplots()
ds.label.isel(time_period=0).plot()
plt.savefig('plots_reeve/K07_map_0.png', bbox_inches='tight')

fig1, ax1 = plt.subplots()
ds.label.isel(time_period=1).plot()
plt.savefig('plots_reeve/K07_map_1.png', bbox_inches='tight')

fig2, ax2 = plt.subplots()
ds.label.isel(time_period=2).plot()
plt.savefig('plots_reeve/K07_map_2.png', bbox_inches='tight')
