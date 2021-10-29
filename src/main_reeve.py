import load_and_preprocess as lp
import bic_and_aic as bic
import plot_tools as pt
import gmm
### plotting tools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
### scikit-learn
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn import manifold
from sklearn import mixture
### xarray, numpy
import xarray as xr
import numpy as np

# load Reeve climatology
ds = lp.load_reeve_climatology()

# select subset (by pressure)
ds = ds.sel(level=slice(300,1000))

# plot mean
dbar = ds.mean(dim="profile", skipna=True)
dbar.prof_CT.plot(y="level", yincrease=False)

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
plt.savefig('allProfs.png',bbox_inches='tight')

# try to get rid of NaN values
CTnn = CT[:, ~np.isnan(CT).any(axis=0)].transpose()
SAnn = SA[:, ~np.isnan(SA).any(axis=0)].transpose()

# scale salinity and temperature
scaled_CT = preprocessing.scale(CTnn)
scaled_SA = preprocessing.scale(SAnn)

# concatenate
Xscaled = np.concatenate((scaled_CT,scaled_SA),axis=0)

# apply pca
pca = PCA(n_components=6)
pca.fit(Xscaled)
Xpca = pca.transform(Xscaled)
total_variance_explained_ = np.sum(pca.explained_variance_ratio_)
print(total_variance_explained_)

bic_mean, bic_std, aic_mean, aic_std = bic.calc_bic_and_aic(Xpca, 20, max_iter=20)
pt.plot_bic_scores("testing", 20, bic_mean, bic_std)
pt.plot_aic_scores("testing", 20, aic_mean, aic_std)

# gmm
myGMM = gmm.train_gmm(Xpca, n_components_selected=10)

# now that GMM has been fit, let's apply it to the original dataset
#labels = myGMM.predict(Xpca)
#posterior_probs = myGMM.predict_proba(Xpca)

# BUT WE NEED TO USE THE ORIGINAL DATASET WITH THE  NANS IN plot_aic_scores
# SO THAT WE WILL KNOW HOW TO PLOT IT. 
