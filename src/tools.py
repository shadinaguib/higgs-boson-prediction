import numpy as np
from implementations import *

def set_flags(method):
    """ Set acronym and flags for the cross validation training and the GridSearch algorithm"""

    if method == least_squares:
        LS=False
        RR=False
        RLR=False
        acronym = 'LS'

    if method in [least_squares_GD, least_squares_SGD, logistic_regression]:
        LS=True
        RR=False
        RLR=False
        if method == least_squares_GD:
            acronym = 'LSGD'
        elif method == least_squares_SGD:
            acronym = 'LSSGD'
        elif method == logistic_regression:
            acronym = 'LR'

    if method == ridge_regression:
        LS=False
        RR=True
        RLR=False
        acronym = 'RR'
        print(acronym)

    if method == reg_logistic_regression:
        LS=False
        RR=False
        RLR=True
        acronym = 'RLR'
    return acronym


def build_poly(x, degree):
    """ Construct the features polynomial expansion and add it to teh dataset """
    for jet in range(4):
        poly = np.ones((len(x), 1))
        for deg in range(1, degree + 1):
            poly = np.c_[poly, np.power(x, deg)]
    return poly
