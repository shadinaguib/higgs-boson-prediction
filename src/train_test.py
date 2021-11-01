import numpy as np
from helpers_p1 import create_csv_submission
from tools import *


def train_jet(train_data_jet, method, jet):
    """ compute the weights using the method on the train dataset"""
    #_,_,_,acronym = set_flags(method)
    acronym = 'RR'
    if jet== 0:
        degree = 9
        lambda_ = 1e-4
        k_fold = 10
        seed = 6
    elif jet == 1:
        degree = 12
        lambda_ = 1e-2
        k_fold = 10
        seed = 6
    elif jet == 2:
        degree = 12
        lambda_ = 1e-3
        k_fold = 10
        seed = 6
    elif jet == 3:
        degree = 12
        lambda_ = 1
        k_fold = 10
        seed = 6

    max_iters = 100
    gamma = 0.1
    #lambda_ = 0.1
    print(lambda_, degree)
    train_pred = train_data_jet[:,1] # 0 is ID, 1 is predictions
    train_features = train_data_jet[:,2:]
    train_features = build_poly(train_features,degree)
    
    # Weight computation using our criterion on the train_set
    if (acronym == 'LSGD'): #train_set[:,0] = actual output values 
        #print("lsgd")
        initial_w = np.random.uniform(low=1e-4, high=0.5, size=train_features.shape[1]) #random initialization of weights
        weights, _ = method(train_pred, train_features, initial_w, max_iters, gamma) 
    
    elif (acronym == 'LSSGD'):
        #print("lssgd")
        initial_w = np.random.uniform(low=1e-4, high=0.5, size=train_features.shape[1]) #random initialization of weights
        weights, _ = method(train_pred, train_features, initial_w, max_iters, gamma) 
    
    elif (acronym == 'LS'):
        #print("ls")
        weights, _ = method(train_pred, train_features) 
    
    elif (acronym == 'RR'):
        #print("rr")
        weights, _ = method(train_pred, train_features, lambda_)
    
    elif (acronym == 'LR'):
        #print("lr")
        initial_w = np.random.uniform(low=1e-4, high=0.5, size=train_features.shape[1]) #random initialization of weights
        weights, _ = method(train_pred, train_features, initial_w, max_iters, gamma)
    
    elif (acronym == 'RLR'):
        #print('rlr')
        initial_w = np.random.uniform(low=1e-4, high=0.5, size=train_features.shape[1]) #random initialization of weights
        weights, _ = method(train_pred, train_features, lambda_, initial_w, max_iters, gamma)
    
    return weights
    
    
    
    
def train(train_data, method):
    """ Call train function for each jet and join in one array"""
    
    weights_0 = train_jet(train_data[0], method, jet=0)
    weights_1 = train_jet(train_data[1], method, jet=1)
    weights_2 = train_jet(train_data[2], method, jet=2)
    weights_3 = train_jet(train_data[3], method, jet=3)
    
    return np.array([weights_0, weights_1, weights_2, weights_3], dtype=object)



def test(test_data, weights):
    """ compute the predictions using trained weights on the test dataset"""
    degree = [9, 12, 12, 12]
    for jet in range(4):
        test_features = test_data[jet][:, 2:]
        test_features = build_poly(test_features, degree[jet])
        yhat = test_features.dot(weights[jet]) > 0.5
        yhat = yhat.astype(int)
        yhat[yhat == 0] = -1 # background at -1 for AI Crowd
        test_data[jet][:, 1] = yhat #Prediction column
    predictions = np.concatenate((test_data[0][:, 0:2], 
                                  test_data[1][:, 0:2], 
                                  test_data[2][:, 0:2], 
                                  test_data[3][:, 0:2]), axis=0)
    return  predictions
        