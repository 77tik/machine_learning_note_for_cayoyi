import copy
import numpy as np

def compute_gradient_matrix(X, y, w, b): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X : (array_like Shape (m,n)) variable such as house size 
      y : (array_like Shape (m,1)) actual value 
      w : (array_like Shape (n,1)) Values of parameters of the model      
      b : (scalar )                Values of parameter of the model      
    Returns
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w. 
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b. 
                                  
    """
    m,n = X.shape
    
    f_wb = X @ w + b

    err = f_wb - y
    dj_dw = (X.T @ err) * (1/m)
    dj_db = np.sum(err) * (1/m)
    return dj_db,dj_dw

def compute_cost_matrix(X, y, w, b, verbose=False):
    """
    Computes the gradient for linear regression 
     Args:
      X : (array_like Shape (m,n)) variable such as house size 
      y : (array_like Shape (m,)) actual value 
      w : (array_like Shape (n,)) parameters of the model 
      b : (scalar               ) parameter of the model 
      verbose : (Boolean) If true, print out intermediate value f_wb
    Returns
      cost: (scalar)                      
    """ 
    m, n = X.shape
    f_wb = X @ w + b
    err = f_wb - y
    cost = np.sum(err**2) * (1/(2*m))
    if verbose: print("f_wb:")
    if verbose: print(f_wb)
    return cost


def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X : (ndarray): Shape (m,n) matrix of examples with multiple features
      w : (ndarray): Shape (n)   parameters for prediction   
      b : (scalar):              parameter  for prediction   
    Returns
      cost: (scalar)             cost
    """
    cost = 0.0
    m,n = X.shape
    for i in range(m):
        
      f_wb = np.dot(X[i],w) + b
      
      cost += (f_wb - y[i])**2
    cost = cost/(2*m)
    return (np.squeeze(cost))

def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X : (ndarray Shape (m,n)) matrix of examples 
      y : (ndarray Shape (m,))  target value of each example
      w : (ndarray Shape (n,))  parameters of the model      
      b : (scalar)              parameter of the model      
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros_like(w)
    dj_db = 0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        for j in range(n):
            dj_dw[j] += (f_wb - y[i]) * (X[i][j]) * (1/m)
        dj_db += f_wb - y[i]
    dj_db = dj_db/m
    return dj_dw,dj_db


def gradient_descent_houses(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X : (array_like Shape (m,n)    matrix of examples 
      y : (array_like Shape (m,))    target value of each example
      w_in : (array_like Shape (n,)) Initial values of parameters of the model
      b_in : (scalar)                Initial value of parameter of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    m,n = X.shape
    hist = {}
    hist["cost"] = []; hist["params"] = []; hist["grads"] = []; hist["iter"] = []; 
    w = copy.deepcopy(w_in)
    b = b_in
    save_interval = np.ceil(num_iters/1000)

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X,y,w,b)
        
        w = w - alpha * dj_dw
        b = w - alpha * dj_db

        if i == 0 or i % save_interval == 0:
            hist["cost"].append(cost_function(X,y,w,b))
            hist["params"].append([w,b])
            hist["grads"].append([dj_dw,dj_db])
            hist["iter"].append(i)

    return w,b,hist

def zscore_normalize_features(X,rtn_ms=False):
    """
    returns z-score normalized X by column
    Args:
      X : (numpy array (m,n)) 
    Returns
      X_norm: (numpy array (m,n)) input normalized by column
    """
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X_norm = (X-mu)/sigma

    if rtn_ms:
        return (X_norm,mu,sigma)
    else:
        return(X_norm)