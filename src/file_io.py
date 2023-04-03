#####################################################################
# Utilities for file input/output and model loading
#####################################################################

import numpy as np
from sklearn import mixture
import xarray as xr
import joblib

#####################################################################
# Import bathymetry file
#####################################################################
def load_bathymetry(file_name="bathy.nc"):

    from netCDF4 import Dataset

    print('file_io.load_bathymetry')

    # import bathymetry file
    #ds = Dataset(file_name, "r", format="NETCDF4")

    # import bathymetry file using xarray
    ds = xr.open_dataset(file_name)
    
    # return the dataset
    return ds

#####################################################################
# Load front
#####################################################################
def load_front(file_name):

    # load a single front
    FRONT = None
    FRONT = np.loadtxt(file_name)

    return FRONT

#####################################################################
# Load SOSE seaice freezing file (winter mean)
#####################################################################
def load_sose_SIfreeze(file_name="physical_fields/SIfreeze_SOSE.nc"):

    # load sea ice files
    ds = xr.open_dataset(file_name)

    # change longitude convention from [0,360] to [-180,180]
    ds = ds.assign_coords({"lon": (((ds.lon + 180) % 360) - 180)})

    # sort by longitude
    ds = ds.sortby("lon")

    return ds

#####################################################################
# Save PCA using joblib
#####################################################################
def save_pca(file_name, pca):

    print('file_io.save_pca')

    # save pca object
    joblib.dump(pca, file_name + '.pkl')

#####################################################################
# Load PCA using joblib
#####################################################################
def load_pca(file_name):

    print('file_io.load_pca')

    # save pca object
    return joblib.load(file_name + '.pkl', 'r')

#####################################################################
# Save GMM as numpy files
#####################################################################
def save_gmm(file_name, gmm):

    print('file_io.save_gmm')

    # save weights, means, and covariances of GMM
    np.save(file_name + '_weights.npy', gmm.weights_, allow_pickle=False)
    np.save(file_name + '_means.npy', gmm.means_, allow_pickle=False)
    np.save(file_name + '_covariances.npy', gmm.covariances_, allow_pickle=False)

#####################################################################
# Load an existing GMM
#####################################################################
def load_gmm(file_name):

    print('file_io.load_gmm')

    # load means and covariances
    means = np.load(file_name + '_means.npy')
    covar = np.load(file_name + '_covariances.npy')

    # reconstruct GMM using means and covariances
    loaded_gmm = mixture.GaussianMixture(n_components = len(means), covariance_type='full')
    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    loaded_gmm.weights_ = np.load(file_name + '_weights.npy')
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar

    return loaded_gmm
