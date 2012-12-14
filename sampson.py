from numpy import *
from numpy.linalg import *
import numpy as np

from algebra import *

# Compute the approximate projection of x0 onto a variety V where V = { x : f(x) = 0 }
# This function returns the difference between x0 and its (approximate) projection onto V
# The parameters are:
# x0  - any point
# f0  = f(x0)
# J0 = Jacobian of f at x0
def firstorder_correction(x0, f0, J0):
    J0 = asarray(J0)
    assert J0.ndim in (1,2)
    if J0.ndim == 1:
        J0 = J0[:,newaxis]  # Treat J0 as a column vector by default
    else:
        assert J0.ndim == 2
    return -squeeze(dots(J0, inv(dot(J0.T, J0)), f0))

# Project x0 onto a variety V where V = { x : f(x) = 0 }
def firstorder_reprojection(x0, f0, J0):
    return x0 + firstorder_correction(x0, f0, J0)

# Compute the norm of the first order deviation of X0 from a variety V
def firstorder_error(x0, f0, J0):
    return sum(square(firstorder_correction(x0, f0, J0)))
