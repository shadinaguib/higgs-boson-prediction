import numpy as np

### Helper functions for Least Squares using gradient descent

def compute_gradient(y, tx, w, sgd=False):
    """Compute the gradient."""
    err = y - tx.dot(w)
    if sgd:
        grad = -tx.T.dot(err)
        return grad, err
  
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_loss(y, tx, w):
    """ Compute the MSE loss"""
    e = y - np.dot(tx,w)
    return calculate_mse(e)



### Helper functions for Logistic Regression using gradient descent

def sigmoid(x):
    """Activation function used for logistic regression"""
    sigma = 1.0 / (1+np.exp(-x))
    return sigma

def compute_loss_grad_logreg(y, tx, w):
    """calculate the cost using negative log likelihood, and the gradient of the loss"""
    yhat = sigmoid(np.dot(tx, w))
    loss = y.T.dot(np.log(yhat)) + (1 - y).T.dot(np.log(1 - yhat))
    grad = tx.T.dot(yhat - y)
                    
    return (loss, grad)



### Helper function for Penalized Logistic Regression using gradient descent

def compute_loss_grad_penalized_logreg(y, tx, w, lambda_):

    loss, grad = compute_loss_grad_logreg(y, tx, w)
    
    loss += lambda_ * (w.T.dot(w)) 
    grad += 2 * lambda_ * w
    
    return loss, grad