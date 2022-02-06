################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
import os
import pickle

import numpy as np
import yaml


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    raise NotImplementedError('One Hot Encoding not implemented')


def write_to_file(path, data):
    """
    Dumps pickled data into the specified relative path.

    Args:
        path: relative path to store to
        data: data to pickle and store
    """
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(train=True):
    """
    Load the data from disk

    Args:
        train: Load training data if true, else load test data

    Returns:
        Tuple:
            Images
            Labels
    """
    directory = 'train' if train else 'test'
    patterns = np.load(os.path.join('./data/', directory, 'images.npz'))['arr_0']
    labels = np.load(os.path.join('./data/', directory, 'labels.npz'))['arr_0']
    return patterns.reshape(len(patterns), -1), labels


def load_config(path):
    """
    Load the configuration from config.yaml

    Args:
        path: A relative path to the config.yaml file

    Returns:
        A dict object containing the parameters specified in the config file
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def generate_k_fold_set(dataset, k=5):
    """
    Creates a generator object to generate k folds for k fold cross validation.

    Args:
        dataset: The dataset to create folds on
        k: The number of folds

    Returns:
        A train and validation fold for each call, up to k times
    """
    X, y = dataset
    if k == 1:
        yield (X, y), (X[len(X):], y[len(y):])
        return

    order = np.random.permutation(len(X))

    fold_width = len(X) // k

    l_idx, r_idx = 0, fold_width

    for i in range(k):
        train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), np.concatenate(
            [y[order[:l_idx]], y[order[r_idx:]]])
        validation = X[order[l_idx:r_idx]], y[order[l_idx:r_idx]]
        yield train, validation
        l_idx, r_idx = r_idx, r_idx + fold_width


def z_score_normalize(X, u=None, sd=None):
    """
    Performs z-score normalization on X.
    f(x) = (x - μ) / σ
        where
            μ = mean of x
            σ = standard deviation of x

    Args:
        X: the data to min-max normalize
        u: the mean to normalize X with
        sd: the standard deviation to normalize X with

    Returns:
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.

    """
    if u is None:
        u = np.mean(X, axis=0)
    if sd is None:
        sd = np.std(X, axis=0)
    return ((X - u) / sd), (u, sd)
