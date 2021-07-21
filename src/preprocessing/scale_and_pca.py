#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 20:06:40 2021

@author: dcjones3
"""

def scale_and_pca(profiles, n_components):
# scale_and_pca(profiles, n_components)
    
    # import modules
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    #import xarray as xr
    import numpy as np
    
    # scale salinity
    X = profiles.prof_S
    scaled_S = preprocessing.scale(X)
    scaled_S.shape
    
    # scale temperature
    X = profiles.prof_T
    scaled_T = preprocessing.scale(X)
    scaled_T.shape
    
    # concatenate 
    Xscaled = np.concatenate((scaled_T,scaled_S),axis=1)
    
    # create PCA object
    pca = PCA(n_components=3)
    
    # fit PCA model
    pca.fit(Xscaled)
    
    # transform input data into PCA representation
    Xpca = pca.transform(Xscaled)
    
    # add PCA values to the profiles Dataset
    #PCA1 = xr.DataArray(Xpca[:,0],dims='profile')
    #PCA2 = xr.DataArray(Xpca[:,1],dims='profile')
    #PCA3 = xr.DataArray(Xpca[:,2],dims='profile')
    
    # calculated total variance explained
    #total_variance_explained_ = np.sum(pca.explained_variance_ratio_) 
    
    return Xpca, pca