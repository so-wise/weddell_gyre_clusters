#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 20:11:47 2021

@author: dcjones3
"""

def calc_bic(Xpca, max_N, max_bic_iter=20):
# calc_bic(Xpca, max_N, max_bic_iter=20)
# returns bic_mean, bic_std
    
    from sklearn import mixture
    import numpy as np
    import random
    
    # initialise, declare variables
    #lowest_bic = np.infty
    bic_scores = np.zeros((2,max_bic_iter))
    
    # loop through the maximum number of classes, estimate BIC
    n_components_range = range(2, max_N)
    bic_iter_range = range(0,max_bic_iter)
    # iterate through all the covariance types (just 'full' for now)
    cv_types = ['full']
    for cv_type in cv_types:
        # iterate over all the possible numbers of components
        for n_components in n_components_range:
            bic_one = []
            # repeat the BIC step for better statistics
            for bic_iter in bic_iter_range:
                # select a new random subset
                rows_id = random.sample(range(0,Xpca.shape[0]-1), 1000)
                Xpca_for_BIC = Xpca[rows_id,:]
                # fit a Gaussian mixture model
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type,
                                              random_state=42)
    
                # uncomment for 'rapid' BIC fitting
                gmm.fit(Xpca_for_BIC)
                # uncomment for 'full' BIC fitting
                #gmm.fit(Xpca)
    
                # append this BIC score to the list
                bic_one.append(gmm.bic(Xpca_for_BIC))
                Xpca_for_BIC = []
    
            # stack the bic scores into a single 2D structure
            bic_scores = np.vstack((bic_scores, np.asarray(bic_one)))
    
    # the first two rows are not needed; they were only placeholders
    bic_scores = bic_scores[2:,:]
    
    # mean values for BIC
    bic_mean = np.mean(bic_scores, axis=1)
    
    # standard deviation for BIC
    bic_std = np.std(bic_scores, axis=1)
    
    return bic_mean, bic_std