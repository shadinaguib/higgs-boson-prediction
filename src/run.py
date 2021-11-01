import numpy as np
from preprocessing import preprocessing
from train_test import train, test
from helpers_p1 import create_csv_submission
from implementations import *


# Chose the method to use
#least_squares_GD, least_squares_SGD, least_squares, ridge_regression, logistic_regression, reg_logistic_regression
method = ridge_regression


# Load the data
print("Loading of the data \n")
DATA_PATH = "../Data/"

train_raw = np.genfromtxt(DATA_PATH+"train.csv", delimiter=',', dtype=None, encoding=None) #dtype=None to get predicions
test_raw = np.genfromtxt(DATA_PATH+"test.csv", delimiter=',', dtype=None, encoding=None) #dtype=None to get predicions


# Preprocessing function returns data where remaining NaN values where replaced by mean, median or zero
# Here we only take the median dataset
print("Preprocessing of the data \n")
_,train_data,_,_, test_data,_ = preprocessing(train_raw, test_raw)


# Compute the trained weights for the corresponding method
print("Training of the weights \n")
weights = train(train_data, method)


# Compute the predictions using trained weights on test data
print("Computation of the test prediction \n")
prediction = test(test_data, weights)


# Create CSV file for submission
print("Creation of the submission file \n")
submission = create_csv_submission(prediction[:, 0], prediction[:,1], "submission")
print("Submission file created under 'submission.csv'\n")
