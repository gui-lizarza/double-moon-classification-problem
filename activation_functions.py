import numpy as np

def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z, alpha=0):
    
    A = np.where(Z >= 0, Z, Z*alpha)
    cache = Z
    
    return A, cache


def tanh(Z):
    
    A = np.tanh(Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z
    return A, cache

def relu_backward(dA, cache, alpha=0):
 
    # derivative of ReLU function
    
    Z = cache
    
    aux = np.ones(shape = dA.shape)
    aux = np.where(Z >= 0, aux, alpha)
    dZ = dA*aux
    
    return dZ

def sigmoid_backward(dA, cache):

    # derivative of sigmoid function
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA*s*(1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def tanh_backward(dA, cache):

    # derivative of hyperbolic tangent function
    
    Z = cache
    
    s = np.tanh(Z)
    dZ = dA*(1 - np.power(s, 2))
    
    assert(dZ.shape == Z.shape)
    
    return dZ
