import numpy as np
from numpy import *
from numpy.linalg import *
import scipy.stats
import unittest

import finite_differences
from algebra import *
from numpy_test import NumpyTestCase,spy

random.seed(123)

# Sensor model:
MEASUREMENT_COV = diag([ .5, .2, .8 ])

# Kinematics model:
KINEMATICS_COV = diag([ .01, .15 ])

# Prior on very first stats
PRIOR_MEAN = array([1., 2.])
PRIOR_COV  = diag([.5, .1])

# Optimization params
MAX_STEPS           = 15
CONVERGENCE_THRESH  = 1e-5   # convergence detected when improvement less than this
INIT_DAMPING        = .1
WINDOW_SIZE         = 4


################################################################################
# Flatten a 2d array of 2d tiles into a 2d array
def flatten2d(A):
    return A.transpose((0,2,1,3)).reshape((A.shape[0]*A.shape[2],
                                           A.shape[1]*A.shape[3]))

################################################################################
# Get the most likely (also mean) measurement vector for state x
def predict_measurement(x):
    return array([ 2.*x[0] + x[1],
                   x[0]*x[0] + x[1]*x[1],
                   x[1]**4 + x[1] + 1.
                   ])

def Jpredict_measurement(x):
    return array([[ 2.       ,  1               ],
                  [ 2*x[0]   ,  2.*x[1]         ],
                  [ 0.       ,  4.*x[1]**3 + 1. ]])


################################################################################
# Get the most likely (also mean) next state for state x
def predict_next(x):
    return array([ norm(x),
                   x[1]
                   ])

def Jpredict_next(x):
    return array([[ x[0]/norm(x) , x[1]/norm(x) ],
                  [ 0.           , 1.           ]])


################################################################################
# Compute the log likelihood of states x given measurements z
def compute_measurement_loglik(xs, zs):
    loglik = 0.
    A = inv(MEASUREMENT_COV)
    for x,z in zip(xs,zs):
        r = predict_measurement(x) - z  # residuals
        loglik -= dots(r, A, r)  # Note that we omit the 1/2 term here
    return loglik

################################################################################
# Compute the log likelihood of states x given kinematics
def compute_kinematic_loglik(xs):
    loglik = 0.
    A = inv(KINEMATICS_COV)
    for i in range(len(xs)-1):
        r = predict_next(xs[i]) - xs[i+1]
        loglik -= dots(r, A, r)   # Note that we omit the 1/2 term here
    return loglik

################################################################################
# Compute the posterior for a set of states given a prior and a set of
# measurements
def compute_logposterior(xs, zs, prior_mean, prior_cov):
    r = xs[0] - prior_mean
    p0 = -dots(r, inv(prior_cov), r)
    p1 = compute_kinematic_loglik(xs)
    p2 = compute_measurement_loglik(xs, zs)
    return p0 + p1 + p2

################################################################################
# Numerically check whether x0 is a local minima by searching a neighbourhood about x0
def delta(i, n, h=1.):
    x = zeros(n)
    x[i] = h
    return x

def is_local_minima(f, x0, h=1e-8):
    assert ndim(x0) == 1
    xlen = shape(x0)[0]
    f0 = f(x0)
    assert isscalar(f0), 'Local minima only valid for scalar-valued functions'

    for i in range(xlen):
        fa = f(x0 - delta(i, xlen, h))
        if fa < f0:
            print '%f < %f !' % (fa,f0)
            return False
        fb = f(x0 + delta(i, xlen, h))
        if fb < f0:
            print '%f < %f !' % (fb,f0)
            return False

    return True

################################################################################
def optimize_window(zs, prior_mean, prior_cov, xs_init):
    num_steps = 0
    converged = False
    damping = INIT_DAMPING

    cost = lambda xs: -compute_logposterior(xs, zs, prior_mean, prior_cov) 
    #cost = lambda xs: -compute_measurement_loglik(xs, zs) 

    xs_cur = xs_init.copy()
    cost_cur = cost(xs_cur)

    while num_steps < MAX_STEPS and damping < 1e+8 and not converged:
        print 'Step %d:   cost=%-10f damping=%f' % (num_steps, cost_cur, damping)

        # Compute J^T * J and J^T * r
        nstates = len(xs_cur)
        A = zeros((nstates, nstates, 2, 2))
        b = zeros((nstates, 2))

        # Add prior term
        A[0,0] = inv(prior_cov)                   # Fisher information matrix
        b[0]   = dot(inv(prior_cov), prior_mean)  # information vector

        for i in range(nstates):
            # Add terms for measurements
            rmsm    =  predict_measurement(xs_cur[i]) - zs[i]
            Jmsm    = Jpredict_measurement(xs_cur[i])
            b[i]   += dot(Jmsm.T, rmsm)
            A[i,i] += dot(Jmsm.T, Jmsm)

            # print 'prediction for state',i
            # print predict_measurement(xs_cur[i])
            # print 'observation for state',i
            # print zs[i]
            # print 'residual for state',i
            # print rmsm
            # print 'Jacobian for state',i
            # print Jmsm

            # Add terms for kinematics
            if i < nstates-1:
                rkin         =  predict_next(xs_cur[i]) - xs_cur[i+1] 
                Jkin         = Jpredict_next(xs_cur[i])
                b[i+1]      += dot(Jkin.T, rkin)
                A[i,   i+1] += dot(Jkin.T, Jkin)
                A[i+1, i  ] += A[i,i+1]

        # Pick a step length
        while damping < 1e+8 and not converged:
            # Apply Levenberg-Marquardt damping
            bflat = b.flatten()
            Adamped = flatten2d(A).copy()
            for i in range(A.shape[0]):
                Adamped[i,i] *= (1. + damping)
    
            # Solve normal equations
            try:
                update = -solve(Adamped, bflat)
            except LinAlgError:
                # Badly conditioned: increase damping and try again
                print '...badly conditioned'
                damping *= 10.
                continue

            # Take gradient step
            xs_next = xs_cur + update.reshape(xs_cur.shape)

            # Compute new cost
            cost_next = cost(xs_next)
            if cost_next < cost_cur:
                # Cost decreased: accept the update
                if cost_cur - cost_next < CONVERGENCE_THRESH or damping < 1e-8:
                    print '  converged due to small improvement'
                    converged = True

                xs_cur = xs_next
                cost_cur = cost_next
                damping *= .1
                num_steps += 1
                break

            else:
                #print '...no improvement'
                # Cost increased: reject the udpate, try another step length
                damping *= 10.

    if converged:
        print 'CONVERGED AFTER %d STEPS' % num_steps
    else:
        print 'DID NOT CONVERGE AFTER %d STEPS (damping=%f)' % (num_steps,damping)

    return xs_cur

################################################################################
# Compute the posterior p(x | z) where x is a state and z is a measurement.
# The likelihood is p(z | x) = Normal(z ; mean, cov)
#    where mean = prediction and cov = inv(z_info)
# The prior p(x) is assumed to be uniform.
# The parameters are as follows.
#   x is the current state
#   prediction is the predicted measurement for the current state
#   J is the jacobian of the prediction function, evaluated at the prediction
#   z is the measurement
#   z_info is the inverse of the covariance of the sensor model (i.e. it is the information matrix for z)
# The function returns (post_info_m, post_info_v), which are the
# parameters for the distribution p(x | z) = Normal(x ; mean, cov)
# where:
#   mean = inv(post_info_m) * post_info_v
#   cov  = inv(post_info_m)
def compute_measurement_posterior(x, prediction, J, z, z_info):
    residual = z - prediction
    post_info_v = dots(J.T, z_information, residual)
    post_info_m = dots(J.T, z_info, J)
    return post_info_v, post_info_m
    

################################################################################
def sample_trajectory(length=10):
    # Construct a ground truth state trajectory
    xs = [ ]
    zs = [ ]
    for i in range(length):
        if i == 0:
            x = random.multivariate_normal(PRIOR_MEAN, PRIOR_COV)
        else:
            x = random.multivariate_normal(predict_next(xs[i-1]), KINEMATICS_COV)
        xs.append(x)
        zs.append(random.multivariate_normal(predict_measurement(x), MEASUREMENT_COV))

    xs = array(xs)
    zs = array(zs)

    return xs, zs
    
################################################################################
def test_optimization():
    TRAJECTORY_LENGTH = 10
    INIT_PERTURBATION = .1

    # Sample states and measurements
    xs_true, zs = sample_trajectory(TRAJECTORY_LENGTH)

    # Pick a starting point
    xs_init = array(xs_true) + random.randn(*shape(xs_true)) * INIT_PERTURBATION

    # Refine the window
    xs_opt = optimize_window(zs, PRIOR_MEAN, PRIOR_COV, xs_init)

    # Evaluate the new states
    cost = lambda x: -compute_logposterior(x.reshape(xs_true.shape), zs, PRIOR_MEAN, PRIOR_COV)
    is_minima = is_local_minima(cost, xs_opt.flatten(), 1e-1)
    if is_minima:
        print 'Estimated window is a minima'
    else:
        print 'Estimated window IS NOT A MINIMA'

    # Report
    print 'True window:'
    print xs_true
    print 'Initial window:'
    print xs_init
    print 'Estimated window:'
    print xs_opt

################################################################################
class WindowTest(NumpyTestCase):
    def setUp(self):
        self.x = array([3., -2.5])

    def testPredictMeasurementsJacobian(self):
        self.assertJacobian(predict_measurement, Jpredict_measurement, self.x)

    def testPredictNextJacobian(self):
        self.assertJacobian(predict_next, Jpredict_next, self.x)

if __name__ == '__main__':
    test_optimization()
    #unittest.main()
