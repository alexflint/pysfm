from numpy import *
from numpy.linalg import *

CAUCHY_SIGMA = .1           # determines the steepness of the Cauchy robustifier near zero
CAUCHY_SIGMA_SQR = CAUCHY_SIGMA * CAUCHY_SIGMA

# Convenience for matrix multiplication
def dots3(A, B, C):
    return dot(A, dot(B, C))

def dots4(A, B, C, D):
    return dot(A, dot(B, dot(C,D)))

#
# SO(3) stuff
#

# Get the skew-symmetric matrix for m
def skew(m):
    return array([[  0,    -m[2],  m[1] ],
                  [  m[2],  0,    -m[0] ],
                  [ -m[1],  m[0],  0.   ]])

# Compute skew(m) * skew(m)
def skewsqr(m):
    a,b,c    = m
    aa,bb,cc = square(m)  # element-wise square
    return array([[ -bb-cc,   a*b,    a*c   ],
                  [  a*b,    -aa-cc,  b*c   ],
                  [  a*c,     b*c,   -aa-bb ]])

# Compute matrix exponential from so(3) to SO(3) (i.e. evaluate Rodrigues)
def SO3_exp(m):
    t = norm(m)
    if t < 1e-8:
        return eye(3)   # exp(0) = I

    a = sin(t)/t
    b = (1. - cos(t)) / (t*t)
    return eye(3) + a*skew(m) + b*skewsqr(m)


#
# Robustifiers
#

# Given a 2x1 reprojection error (in raw pixel coordinates), evaluate
# a robust error function and return a scalar cost.
def cauchy_cost_from_reprojection_error(x):
    x = asarray(x)
    return log(1. + (x[0]*x[0] + x[1]*x[1]) / CAUCHY_SIGMA_SQR)

# Given a 2x1 reprojection error (in raw pixel coordinates), evaluate
# a robust error function and return:
#   -- the 2x1 residual vector
#   -- the 2x2 Jacobian of that residual
def cauchy_residual_from_reprojection_error(x):
    x = asarray(x)
    rr = x[0]*x[0] + x[1]*x[1]
    if rr < 1e-8:   # within this window the residual is well-approximated as linear
        J_near_zero = eye(2) / CAUCHY_SIGMA
        r_near_zero = x / CAUCHY_SIGMA
        return r_near_zero, J_near_zero

    r = sqrt(rr)
    e = sqrt(log(1. + rr / CAUCHY_SIGMA_SQR))
    xx = outer(x, x)
    residual = x * e / r
    Jresidual = xx / (r*e*(rr + CAUCHY_SIGMA_SQR)) + (r*eye(2) - xx/r) * e/rr
    return residual, Jresidual



# Triangulate a 3D point from a set of observations by cameras with
# fixed parameters. Uses least squares on the algebraic error.
def triangulate_algebraic_lsq(K, Rs, ts, msms):
    A = empty((len(Rs)*2, 3))
    b = empty(len(Rs)*2)
    msms = asarray(list(msms))

    for i in range(len(Rs)):
        b[i*2]   = dot(msms[i,0] * K[2] - K[0], ts[i])
        b[i*2+1] = dot(msms[i,1] * K[2] - K[1], ts[i])
        A[i*2]   = dot(K[0] - msms[i,0] * K[2], Rs[i])
        A[i*2+1] = dot(K[1] - msms[i,1] * K[2], Rs[i])

    x, residuals, rank, sv = lstsq(A, b)
    return x
