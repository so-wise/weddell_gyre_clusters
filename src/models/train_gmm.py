#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 20:16:06 2021

@author: dcjones3
"""

def train_gmm(profiles, Xpca, n_components_selected, random_state=42):
# train_gmm(profiles, Xpca, n_components_selected, random_state=42)
# returns profile object with labels sand posterior probabilities
    
    #import numpy as np
    from sklearn import mixture
    import xarray as xr
    
    # establish gmm
    best_gmm = mixture.GaussianMixture(n_components=n_components_selected,
                                       covariance_type='full',
                                       random_state=random_state)
    
    # fit this GMM
    best_gmm.fit(Xpca)
    
    # check to make sure that n_comp is as expected
    #n_comp = best_gmm.n_components
    
    # select colormap
    #colormap = plt.get_cmap('tab10', n_comp)
    
    # assign class labels ("predict" the class using the selected GMM)
    labels = best_gmm.predict(Xpca)
    
    # find posterior probabilities (the probabilities of belonging to each class)
    posterior_probs = best_gmm.predict_proba(Xpca)
    
    # maximum posterior probability (the class is assigned based on this value)
    #max_posterior_probs = np.max(posterior_probs,axis=1) 
    
    # put the labels and maximum posterior probabilities back in original dataframe
    #df.insert(3,'label',labels,True)
    #df.insert(4,'max posterior prob',max_posterior_probs,True) 
    
    # print out best_gmm parameters
    #posterior_probs.shape
    
    # convert labels into xarray format
    xlabels = xr.DataArray(labels, coords=[profiles.profile], dims='profile')
    
    # convert posterior probabilities into xarray format
    gmm_classes = [b for b in range(0,n_components_selected,1)]
    xprobs = xr.DataArray(posterior_probs, 
                          coords=[profiles.profile, gmm_classes], 
                          dims=['profile', 'CLASS'])
    
    # add label DataArray to Dataset
    profiles = profiles.assign({'label':xlabels})
    profiles = profiles.assign({'posteriors':xprobs})
    
    return profiles