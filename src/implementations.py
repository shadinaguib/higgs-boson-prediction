
## Implementations of functions for project 1 methods ##

import numpy as np
from costs import *


"""
For all functions in file implementations.py :

param y:         actual output value
param tx:        data samples
param w:         initial weight vector
param max_iters: number of steps to run for gradient descent
param gamma:     step size for gradient descent
param lambda_:   regularization parameter for ridge regression and regularized logistic regression

return (w, loss): w is the last weight vector of the method, loss is the corresponding cost function value
"""

### Least Squares with Gradient Descent ###

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Selects optimal weights based on the Gradient descent method through the minimization of the MSE cost function """

    tol = 1e-4
    w_old = initial_w
    loss_old = 0
    for i in range(int(max_iters)):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w_old)
        loss = calculate_mse(err)
        # gradient descent update
        w = w_old - gamma * grad

        # Stops GD if cost/loss reduction is less than say 0.01% from the previous loss, precision TBD
        if i>1:
            improvement = abs(loss-loss_old) / abs(loss_old)
            if improvement < tol:
                break
            else:
                w_old = w
                loss_old = loss

    return (w, loss)



### Least Squares with Stochastic Gradient Descent ###

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Computes the optimal weights based on the Stochastic Gradient descent method through the minimization of the MSE cost function and with a batch size of 1."""
    tol = 1e-4
    w_old = initial_w
    loss_old = 0
    for i in range(int(max_iters)):
        losses = []
        for n in range(tx.shape[0]):
            # Compute a stochastic gradient with minibatch size of 1
            grad,_ = compute_gradient(y[n], tx[n,:], w_old, sgd=True)

            # update w through the stochastic gradient update
            w = w_old - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # Keep track of the losses during an epoch
            losses.append(loss)
        # mean loss of an epoch over all the dataset
        loss = np.mean(losses)

        # Stops GD if cost/loss reduction is less than say 0.01% from the previous loss, precision TBD
        if i>1:
            improvement = abs(loss-loss_old) / abs(loss_old)
            if improvement < tol:
                break
            else:
                w_old = w
                loss_old = loss

    return (w, loss)




### Least Squares ###

def least_squares(y, tx):
    """ Computes directly the optimal weight of the least squares solution """

    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)

    return (w, loss)




### Ridge Regression ###

def ridge_regression(y, tx, lambda_):
    """ Computes directly the optimal weight with the Least Squares Method and a L2-Regularization, called Ridge Regression """

    penalty = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + penalty
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w) # MSE loss

    return (w, loss)



### Logistic Regression using Gradient Descent ###


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Computes the optimal weights in order to have a binary classification prediction with the logistic regression method """
    tol = 1e-4
    w_old = initial_w
    loss_old = 0
    for i in range(int(max_iters)):
        loss, grad = compute_loss_grad_logreg(y, tx, w_old)
        w = w_old - gamma * grad

        # Stops GD if cost/loss reduction is less than say 0.01% from the previous loss, precision TBD
        if i>1:
            improvement = abs(loss-loss_old) / abs(loss_old)
            if improvement < tol:
                break
            else:
                w_old = w
                loss_old = loss

    return (w, loss)




### Regularized Logistic Regression ###

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Computes the optimal weights in order to have a binary classification prediction with the logistic regression method and L2-norm regularization """
    tol = 1e-4
    w_old = initial_w
    loss_old = 0
    for i in range(int(max_iters)):
        loss, grad = compute_loss_grad_penalized_logreg(y, tx, w_old, lambda_)
        w = w_old - gamma * grad

        # Stops GD if cost/loss reduction is less than say 0.01% from the previous loss, precision TBD
        if i>1:
            improvement = abs(loss-loss_old) / abs(loss_old)
            if improvement < tol:
                break
            else:
                w_old = w
                loss_old = loss

    return (w, loss)
