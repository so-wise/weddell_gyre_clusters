#####################################################################
# Utilities for model selection (e.g. BIC, AIC)
#####################################################################

# import modules
from sklearn import mixture
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import random

#####################################################################
# Calculate BIC and AIC (and silhouette score) 
#####################################################################
def calc_bic_and_aic(Xpca, max_N, max_iter=20):
# calc_bic_and_aic(Xpca, max_N, max_iter=20)
# returns bic_mean, bic_std, aic_mean, aic_std

    # start message
    print('bic_and_aic.calc_bic_and_aic')
    print('--- this may take some time ---')

    # initialize, declare variables
    bic_scores = np.zeros((2,max_iter))
    aic_scores = np.zeros((2,max_iter))
    silhouette_scores = np.zeros((2,max_iter))

    # loop through the maximum number of classes, estimate BIC
    n_components_range = range(2, max_N)
    iter_range = range(0,max_iter)

    # iterate through all the covariance types (just 'full' for now)
    cv_types = ['full']

    # loop through cv_types, components, and iterations
    for cv_type in cv_types:
        # iterate over all the possible numbers of components
        for n_components in n_components_range:
            bic_one = []
            aic_one = []
            sil_one = []
            # repeat the BIC step for better statistics
            for bic_iter in iter_range:
                # select a new random subset
                rows_id = random.sample(range(0,Xpca.shape[0]-1), 1000)
                Xpca_for_BIC = Xpca[rows_id,:]
                # fit a Gaussian mixture model
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type,
                                              random_state=42)

                # uncomment for 'rapid' BIC fitting
                gmm.fit(Xpca_for_BIC)

                # append this BIC score to the list
                bic_one.append(gmm.bic(Xpca_for_BIC))
                aic_one.append(gmm.aic(Xpca_for_BIC))
                
                # get labels and silhouette score
                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                cluster_labels_tmp = gmm.predict(Xpca_for_BIC)
                sil_tmp = silhouette_score(Xpca_for_BIC, cluster_labels_tmp)
                sil_one.append(sil_tmp)
                
                # clear array for the next iteration
                Xpca_for_BIC = []

            # stack the bic scores into a single 2D structure
            bic_scores = np.vstack((bic_scores, np.asarray(bic_one)))
            aic_scores = np.vstack((aic_scores, np.asarray(aic_one)))
            silhouette_scores = np.vstack((silhouette_scores, np.asarray(sil_one)))

    # the first two rows are not needed; they were only placeholders
    bic_scores = bic_scores[2:,:]
    aic_scores = aic_scores[2:,:]
    silhouette_scores = silhouette_scores[2:,:]

    # mean values for BIC, AIC, and silhouette
    bic_mean = np.mean(bic_scores, axis=1)
    aic_mean = np.mean(aic_scores, axis=1)
    silhouette_mean = np.mean(silhouette_scores, axis=1)

    # standard deviation for BIC, AIC, and silhouette
    bic_std = np.std(bic_scores, axis=1)
    aic_std = np.std(aic_scores, axis=1)
    silhouette_std = np.std(silhouette_scores, axis=1)

    return bic_mean, bic_std, aic_mean, aic_std, silhouette_mean, silhouette_std
