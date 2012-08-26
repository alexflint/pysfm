from numpy import *
from numpy.linalg import *

def compute(X, ndims):
    X = asarray(X)
    u,s,v = svd(X, full_matrices=False)
    return v[:ndims]

def project(X, subspace):
    return dot(X, subspace.T)

