""" Gather functions useful for the cross-validation"""

import numpy as np
from implementations import *
from tools import *


def create_k_indices(dataset, k_fold, seed):
    """Create k_fold array of indexes to split the dataset for cross validation"""

    nbr_samples = dataset.shape[0]
    fold_size = int(nbr_samples / k_fold)
    np.random.seed(seed)
    idx = np.random.permutation(nbr_samples)
    k_idx = []
    for k in range(k_fold):
        idx_array = np.array(idx[k*fold_size : (k+1)*fold_size])

        # If some idx are still missing
        if k == k_fold-1:
            nbr_missing =  nbr_samples - k_fold*fold_size
            while nbr_missing > 0:
                idx_array = np.append(idx_array, idx[-nbr_missing])
                nbr_missing -= 1

        k_idx.append(idx_array)

    return k_idx


def cross_validation(dataset, acronym, method, seed=6,  k_fold=10, deg=12, max_iters=100, gamma=0.1, lambda_=0.1):
    """ Main function for crossvalidation.  """
    # Create the indices lists for k-flod split of teh dataset
    k_idx = create_k_indices(dataset, k_fold, seed)
    accuracies = []
    np.random.seed(seed)

    for k in range(k_fold):
        # Test set and train set for kth cross validation step
        validation_set = dataset[k_idx[k-1]] # only 1/k_flod of the rows
        train_set = np.delete(dataset, k_idx[k-1], axis=0) # all other rows

        # Polynomial expansion of the training and test set's features (without the prediction column)
        validation_features = build_poly(validation_set[:,1:], deg)
        train_features = build_poly(train_set[:,1:],deg)

        # Weight computation using our criterion on the train_set
        if (acronym == 'LSGD'): #train_set[:,0] = actual output values
            initial_w = np.random.uniform(low=1e-4, high=0.5, size=train_features.shape[1]) #random initialization of weights
            weights, _ = method(train_set[:,0], train_features, initial_w, max_iters, gamma)
        elif (acronym == 'LSSGD'):
            initial_w = np.random.uniform(low=1e-4, high=0.5, size=train_features.shape[1]) #random initialization of weights
            weights, _ = method(train_set[:,0], train_features, initial_w, max_iters, gamma)
        elif (acronym == 'LS'):
            weights, _ = method(train_set[:,0], train_features)
        elif (acronym == 'RR'):
            weights, _ = ridge_regression(train_set[:,0], train_features, lambda_)
        elif (acronym == 'LR'):
            initial_w = np.random.uniform(low=1e-4, high=0.5, size=train_features.shape[1]) #random initialization of weights
            weights, _ = method(train_set[:,0], train_features, initial_w, max_iters, gamma)
        elif (acronym == 'RLR'):
            initial_w = np.random.uniform(low=1e-4, high=0.5, size=train_features.shape[1]) #random initialization of weights
            weights, _ = method(train_set[:,0], train_features, lambda_, initial_w, max_iters, gamma)

        # Computation of prediction with trained weights on validation set
        yhat = validation_features.dot(weights) > 0.5

        # Accuracy computation for one fold and store to compute mean after all folds
        accuracy = np.sum(yhat == validation_set[:,0]) / len(yhat) #validation[:,0] = actual output values
        accuracies.append(accuracy)
    return (np.mean(accuracies))
