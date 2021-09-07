#####################################################################
# Utilities for file input/output and model loading
#####################################################################

import numpy as np
from sklearn import mixture

#####################################################################
# Save GMM as numpy files
#####################################################################
def save_gmm(file_name, gmm):

    # save weights, means, and covariances of GMM
    np.save(file_name + '_weights.npy', gmm.weights_, allow_pickle=False)
    np.save(file_name + '_means.npy', gmm.means_, allow_pickle=False)
    np.save(file_name + '_covariances.npy', gmm.covariances_, allow_pickle=False)

#####################################################################
# Load an existing GMM
#####################################################################
def load_gmm(file_name):

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
