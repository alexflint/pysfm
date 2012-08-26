import numpy as np

# Project a homogeneous vector or matrix. In the latter case each
# *row* will be interpreted as a vector to be projected.
def pr(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:-1] / x[-1]
    elif x.ndim == 2:
        return x[:,:-1] / x[:,[-1]]
    else:
        raise Exception, 'Cannot pr() an array with %d dimensions' % x.ndim

# Unproject a vector or matrix. In the latter case each *row* will be
# interpreted as a separate vector.
def unpr(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return np.hstack((x, 1.))
    elif x.ndim == 2:
        return np.hstack((x, np.ones((len(x), 1))))
    else:
        raise Exception, 'Cannot unpr() an array with %d dimensions' % x.ndim

# Project a point through a homogeneous transformation, projecting and
# unprojecting automatically if necessary
# For 100k produces this is a 90x speedup over iterating over the rows
def prdot(H, X):
    H = np.asarray(H)
    X = np.asarray(X)
    assert np.ndim(H) == 2, 'The shape of H was %s' % str(H.shape)
    if X.ndim == 1:
        assert len(X) == np.size(H,1)-1, \
            'H.shape was %s, X.shape was %s' % (str(H.shape), str(X.shape))
        return pr(np.dot(H, unpr(X)))
    elif X.ndim == 2:
        assert np.size(X,1) == np.size(H,1)-1, \
            'H.shape was %s, X.shape was %s' % (str(H.shape), str(X.shape))
        return pr(np.dot(unpr(X), H.T))

# Multiple an arbitrary number of matrices with np.dot.
# Surely there is a way in numpy to do this conveniently but I haven't found it
def dots(*m):
    return reduce(np.dot, m)

# Compute the sum of squared elements
def ssq(x):
    return np.dot(x,x)

# Compute the skew-symmetric matrix for m
def skew(m):
    m = np.asarray(m)
    assert m.shape == (3,)
    return np.array([[  0,    -m[2],  m[1] ],
                     [  m[2],  0,    -m[0] ],
                     [ -m[1],  m[0],  0.   ]])

