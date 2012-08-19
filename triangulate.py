import numpy as np
from algebra import *

# Triangulate a 3D point from a set of observations by cameras with
# fixed parameters. Uses least squares on the algebraic error.
def algebraic_lsq(K, Rs, ts, msms):
    A = np.empty((len(Rs)*2, 3))
    b = np.empty(len(Rs)*2)
    msms = np.asarray(msms)

    for i in range(len(Rs)):
        b[i*2]   = dots(msms[i,0] * K[2] - K[0], ts[i])
        b[i*2+1] = dots(msms[i,1] * K[2] - K[1], ts[i])
        A[i*2]   = dots(K[0] - msms[i,0] * K[2], Rs[i])
        A[i*2+1] = dots(K[1] - msms[i,1] * K[2], Rs[i])

    x, residuals, rank, sv = np.linalg.lstsq(A, b)
    return x
