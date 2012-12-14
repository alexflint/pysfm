import numpy as np
from numpy import *
from numpy.linalg import *

################################################################################
# Make a spherical gaussian proposal distribution
def gaussian_proposal(shape, sigma):
    return lambda x: x + random.randn(shape)*sigma

################################################################################
# Sample from a PDF using the Metropolis-Hastings algorithm
def sample_metropolis_hastings(pdf, x0, proposal=None, nsteps=10):
    if proposal is None:
        assert isscalar(x0) or isinstance(x0, ndarray), \
            'For non-scalar, non-array starting points you must give a proposal sampler'
        proposal = gaussian_proposal(shape(x0))

    xcur = x0
    pcur = pdf(xcur)
    assert pcur > 1e-8, 'PDF must be greater than zero at starting point'

    naccept = 0
    for i in range(nsteps):
        xnext = proposal(xcur)
        pnext = pdf(xnext)
        assert pnext >= 0. and pnext <= 1.

        if pnext >= pcur or pnext >= random.rand()*pcur:
            xcur = xnext
            pcur = pnext
            naccept += 1

    print 'Halted with %d accepts' % naccept

    return xcur
    
################################################################################
# Sample from a PDF using the Metropolis-Hastings algorithm
def sample_many(pdf, x0, proposal=None, nsteps=10, nsamples=1):
    xs = []
    for i in range(nsamples):
        xs.append(sample_metropolis_hastings(pdf, x0, proposal, nsteps))

    # Heuristically decide whether to wrap the results in a numpy array
    if isinstance(x0, ndarray):
        return array(xs)
    else:
        return xs

################################################################################
# Test apparatus
def tri(x):
    if x < -1:
        return 0.
    elif x < 0:
        return 1.+x
    elif x <= 1:
        return 1.-x
    else:
        return 0.

def tri_2d(x):
    assert shape(x) == (2,), 'x='+str(x)
    return tri(x[0]) * tri(x[1])

def sample_tri(nsteps, nsamples=1):
    propose = gaussian_proposal(.4)
    return sample_many(tri, 0., propose, nsteps, nsamples)

def sample_tri_2d(nsteps, nsamples=1):
    propose = gaussian_proposal_2d(.4)
    return sample_many(tri_2d, zeros(2), propose, nsteps, nsamples)
