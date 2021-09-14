#####################################################################
# Utilities for training and applying models
#####################################################################

#import numpy as np
from sklearn import mixture
import numpy as np
import xarray as xr

#####################################################################
# Train GMM
#####################################################################
def train_gmm(Xtrain, n_components_selected, random_state=42):
# train_gmm(Xtrain, n_components_selected, random_state=42)
# returns gmm

    print('gmm.train_gmm')

    # establish gmm
    gmm = mixture.GaussianMixture(n_components=n_components_selected,
                                  covariance_type='full',
                                  random_state=random_state)

    # fit this GMM using the training data in PC space
    gmm.fit(Xtrain)

    # return the gmm object, which has now been trained
    return gmm

#####################################################################
# Apply GMM
#####################################################################
def apply_gmm(profiles, Xpca, gmm, n_components_selected, random_state=42):
# train_gmm(profiles, Xpca, n_components_selected, random_state=42)
# returns gmm

    print('gmm.apply_gmm')

    # assign class labels ("predict" the class using the selected GMM)
    labels = gmm.predict(Xpca)

    # find posterior probabilities (the probabilities of belonging to each class)
    posterior_probs = gmm.predict_proba(Xpca)

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

#####################################################################
# Get mean and standard deviation (class statistics)
#####################################################################
def calc_class_stats(profiles):

    print('gmm.calc_class_stats')

    # create grouped object using the labels
    grouped = profiles.groupby("label")

    # class means and standard deviations
    class_means = grouped.mean()
    class_stds = grouped.std()

    return class_means, class_stds

#####################################################################
# Calculate the i-metric
#####################################################################
def calc_i_metric(profiles):

    print('gmm.calc_i_metric')

    # first, get 1D dataframe
    df1D = profiles.isel(depth=0)

    # declare variables
    i_metric = np.zeros(df1D.profile.size)
    a_b = np.zeros((df1D.profile.size,2))

    # loop through the profiles, calculate the i_metric for each one
    for i in range(df1D.profile.size):
          i_metric[i], a_b[i,:] = get_i_metric(df1D.posteriors[i, :].values.tolist())

    # convert i_metric numpy array to xarray DataArray
    i_metric = xr.DataArray(i_metric, coords=[profiles.profile], dims='profile')

    # add i_metric DataArray to Dataset
    df1D = df1D.assign({'i_metric':i_metric})

    # return 1D dataframe with i-i_metric
    return df1D

#####################################################################
# Contains the definition of the i-metric
#####################################################################
def get_i_metric(posterior_prob_list):

    # first, sort the posterior probability list
    sorted_posterior_list = sorted(posterior_prob_list)

    # I_metric = 1 - (max_probability - runner_up)
    ic_metric = 1 - (sorted_posterior_list[-1] - sorted_posterior_list[-2])

    # list the runner-up class (next most likely)
    runner_up_label = posterior_prob_list.index(sorted_posterior_list[-2])

    # what the class with the maximum probability?
    label = posterior_prob_list.index(sorted_posterior_list[-1])

    # return the i-metric, label, and runner-up label
    return ic_metric, np.array([label, runner_up_label]) # np.sort()
