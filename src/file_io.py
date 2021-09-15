#####################################################################
# Utilities for file input/output and model loading
#####################################################################

import numpy as np
from sklearn import mixture
import joblib

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
def load_pca(file_name, pca):

    print('file_io.load_pca')

    # save pca object
    joblib.load(pca, file_name + '.pkl')

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
