import numpy as np
from numpy import *
from numpy.linalg import *

from algebra import *
from lie import SO3
import mcmc
import beliefs
import fundamental
import sampson
import finite_differences

K = fundamental.K

SENSOR_COV  = eye(2)
SENSOR_INFO = inv(SENSOR_COV)

def pmat(R, t):
    return hstack((R, t[:,newaxis]))

def sampson_error(K, R, t, x0, x1):
    return fundamental.residual(K, R, t, x0, x1) ** 2

def se3_chart(Rt, v):
    return ( dot(Rt[0], SO3.exp(v[:3])), Rt[1] + v[3:] )

def fmat_likelihood(K, R, t, xs0, xs1):
    loglik = 0.
    for x0,x1 in zip(xs0, xs1):
        loglik -= sampson_error(K, R, t, x0, x1)
    return exp(loglik)

def fmat_likelihood_charted(K, R0, t0, v, xs0, xs1):
    R,t = se3_chart((R0, t0), v)
    p = fmat_likelihood(K, R, t, xs0, xs1)
    #print '\nProposal:'
    #print v
    #print 'Charted proposal:'
    #print pmat(R,t)
    #print 'Likelihood:',p
    return p

def sample_from_likelihood():
    R_true, t_true, xs0, xs1 = fundamental.setup_test_problem()
    xs0 = unpr(xs0)[:5]
    xs1 = unpr(xs1)[:5]
    Rt_true = (R_true, t_true)

    print 'True:'
    print pmat(R_true, t_true)

    pdf = lambda v: fmat_likelihood_charted(K, Rt_true[0], Rt_true[1], v, xs0, xs1)
    proposal = mcmc.gaussian_proposal(6, .1)
    samples = mcmc.sample_many(pdf, zeros(6), proposal, 500, 10)

    mean = samples.mean(axis=0)
    stddev = samples.std(axis=0)

    print 'Empirical mean:',
    print mean

    print 'Empirical standard deviation:',
    print stddev

# Compute the algebraic cost
def algebraic_cost(F, x0, x1):
    return dots(x1, F, x0)

# Compute the jacobian of the algebraic cost w.r.t. (x0,x1)
def Jalgebraic_cost_x(F, x0, x1):
    return concatenate((dot(F.T,x1), dot(F,x0)))

# Compute the correction to be added to x0 and x1 to reduce the
# aglebraic cost to zero on a first order approximation.
def correction(F, x0, x1):
    x = concatenate((x0,x1))
    f = algebraic_cost(F, x0, x1)
    J = Jalgebraic_cost(F, x0, x1)
    dev = sampson.firstorder_deviation(x, f, J)
    return (dev[:3], dev[3:])

# Compute the the sampson reprojection for x0 and x1
def reprojection(F, x0, x1):
    x = concatenate((x0,x1))
    f = algebraic_cost(F, x0, x1)
    J = Jalgebraic_cost_x(F, x0, x1)
    corr = sampson.firstorder_reprojection(x, f, J)
    return corr[:3], corr[3:]

# Compute the the sampson reprojection for x0 and x1
def report_reprojection(F, x0, x1):
    x = concatenate((x0,x1))
    f = algebraic_cost(F, x0, x1)
    J = Jalgebraic_cost_x(F, x0, x1)
    corr = sampson.firstorder_reprojection(x, f, J)
    xx0, xx1 = corr[:3], corr[3:]
    print 'algebraic cost before: %10f' % algebraic_cost(F, x0, x1)
    print 'algebraic cost after: %10f' % algebraic_cost(F, xx0, xx1)
    return xx0, xx1

def compute_posterior(x0, x1, R_mean, t_mean):
    x = concatenate((x0,x1))

    Fmat = lambda R,t: fundamental.make_fundamental(eye(3), R, t)
    f_alg = lambda R,t: algebraic_cost(Fmat(R, t), x0, x1)
    J_alg = lambda R,t: Jalgebraic_cost_x(Fmat(R, t), x0, x1)

    # sampson errors
    E_sampson = lambda R,t: ssq(sampson.firstorder_deviation(x, f_alg(R,t), J_alg(R,t)))
    E_sampson_charted = lambda v: E_sampson(se3_chart((R_mean, t_mean), v))

    # sampson reprojections
    sampson_prediction = lambda (R,t): sampson.firstorder_reprojection(x, f_alg(R,t), J_alg(R,t))
    sampson_prediction_charted = lambda v: sampson_prediction(se3_chart((R_mean,t_mean), v))

    # prediction jacobian
    Jprediction = finite_differences.numeric_jacobian(sampson_prediction_charted,
                                                      zeros(6))

    # compute likelihood in terms of R and t
    prediction = sampson_prediction((R_mean, t_mean))
    print prediction
    print Jprediction
    print x
    print SENSOR_INFO
    v,L = beliefs.compute_likelihood(zeros(6), prediction, Jprediction, x, eye(6))

    print 'R_mean:'
    print R_mean
    print 't_mean'
    print t

    # compute mean and variance
    print 'v'
    print v.round(2)
    print 'L'
    print L.round(2)
    mean,cov = beliefs.normal_to_invnormal(v,L)
    print 'mean:'
    print mean
    print 'cov:'
    print cov

def experiment():
    

    
    
################################################################################
from numpy_test import NumpyTestCase
    
class FmatTest(NumpyTestCase):
    def setUp(self):
        pass

    def test_algebraic_cost(self):
        x0 = arange(3)
        x1 = arange(5,8)
        F = arange(9).reshape((3,3))

        f = lambda x: algebraic_cost(F, x[:3], x[3:])
        Jf = lambda x: Jalgebraic_cost_x(F, x[:3], x[3:])
        self.assertJacobian(f, Jf, concatenate((x0,x1)), verbose=True)
        
if __name__ == '__main__':
    import unittest
    unittest.main()
