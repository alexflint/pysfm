import numpy as np
from numpy import *
from numpy.linalg import *
import scipy.integrate
import itertools

from numpy_test import NumpyTestCase
import finite_differences
from algebra import *
from geometry import *

LOG_MIN_REPRESENTABLE_FLOAT = -300 * log(10)   # TODO: do this exactly


def is_positive_definite(A):
    evals = eigvals(A)
    return np.all(evals > 0)

################################################################################
def integrate2d(f, lower, upper):
    assert shape(lower) == (2,)
    assert shape(upper) == (2,)
    y,err = scipy.integrate.dblquad(lambda y,x: f(array([x,y])),
                                    lower[0],
                                    upper[0],
                                    lambda x: lower[1],
                                    lambda x: upper[1])
    return y

################################################################################
# Compute the logarithm of the determinant of a symmetric positive
# definite matrix in a numerically stable way
def log_det_covariance(cov):
    return 2. * np.sum(log(diag(cholesky(cov))))

################################################################################
# Compute the logarithm of the determinant of the inverse of a
# information matrix. (Actually is applicable to any symmetric positive definite matrix)
def log_det_inv_infomat(L):
    return 2. * np.sum(log(diag(inv(cholesky(L)))))

################################################################################
# Evaluate the normal distribution at x
def evaluate_normal(x, mean, cov):
    x = asarray(x)
    mean = asarray(mean)
    cov = asarray(cov)

    assert x.ndim == 1
    assert mean.ndim == 1
    assert cov.ndim == 2
    assert x.shape == mean.shape
    assert cov.shape == (x.size, x.size)

    k = size(x)
    expterm = -.5 * dots(x-mean, inv(cov), x-mean)
    if expterm < LOG_MIN_REPRESENTABLE_FLOAT:
        return 0.  # Cannot do better than this unfortunately

    #denom = (2. * pi) ** (k/2.) * sqrt(det(cov))
    #return exp(expterm) / denom

    log_denom = .5*k*log(2. * pi) + .5*log_det_covariance(cov)
    return exp(expterm - log_denom)

################################################################################
# Evaluate the inverse normal distribution at x
def evaluate_invnormal(x, v, L):
    x = asarray(x)
    v = asarray(v)
    L = asarray(L)

    assert x.ndim == 1
    assert v.ndim == 1
    assert L.ndim == 2
    assert x.shape == v.shape
    assert L.shape == (x.size, x.size)

    k = size(x)
    expterm_var = -.5 * dots(x, L, x) + dot(x, v) # This term depends on x
    expterm_const = -.5 * dots(v, inv(L), v)      # This term does not depend on x
    expterm = expterm_var + expterm_const
    if expterm < LOG_MIN_REPRESENTABLE_FLOAT:
        return 0.  # Cannot do better than this unfortunately

    #denom = (2. * pi) ** (k/2.) * sqrt(det(inv(L)))
    #return exp(expterm) / denom

    log_denom = .5*k*log(2. * pi) + .5*log_det_inv_infomat(L)
    return exp(expterm - log_denom)
    


################################################################################
# Sample from the normal distrubtion
def sample_normal(mean, cov):
    return random.multivariate_normal(mean, cov)

################################################################################
# Sample from the inverse normal distrubtion
def sample_invnormal(info_vec, info_mat):
    cov = inv(info_mat)
    return random.multivariate_normal(dot(cov, info_vec), cov)

################################################################################
# Marginalize parameters out of the normal distribution
def marginalize_normal(mean, cov, params_to_keep):
    mean = asarray(mean)
    cov = asarray(cov)
    assert mean.ndim == 1
    assert cov.ndim == 2
    assert cov.shape == (mean.size, mean.size)

    # check the params - tuples are interpreted differently to lists
    params_to_keep = list(params_to_keep)

    # marginalize
    marg_mean = mean[ params_to_keep ]

    marg_cov = cov[ params_to_keep ].T[ params_to_keep ].T
    return marg_mean, marg_cov

################################################################################
# Marginalize parameters out of the inverse normal distribution
def marginalize_invnormal(v, L, params_to_keep):
    v = asarray(v)
    L = asarray(L)
    assert v.ndim == 1
    assert L.ndim == 2
    assert L.shape == (v.size, v.size)
    
    # make a mask
    paramset = set(params_to_keep)
    keep_mask = array([ i in paramset for i in range(len(v)) ])
    
    # pull out the pieces
    Laa = L[  keep_mask  ].T[  keep_mask  ].T
    Lab = L[  keep_mask  ].T[ ~keep_mask  ].T
    Lba = L[ ~keep_mask  ].T[  keep_mask  ].T
    Lbb = L[ ~keep_mask  ].T[ ~keep_mask  ].T
    va  = v[  keep_mask  ]
    vb  = v[ ~keep_mask  ]

    # this is simply the schur compliment
    Lbb_inv = inv(Lbb)
    marg_L = Laa - dots(Lab, Lbb_inv, Lba)
    marg_v = va  - dots(Lab, Lbb_inv, vb)

    return marg_v, marg_L


################################################################################
# Convert from normal to inverse normal parametrisation
def normal_to_invnormal(mean, cov):
    return dot(inv(cov), mean), inv(cov)

################################################################################
# Convert from inverse normal to normal parametrisation
def invnormal_to_normal(v, L):
    return dot(inv(L), v), inv(L)

################################################################################
# Compute product of normal distributions in ordinary parametrisation
def product_of_normals(mean1, cov1, mean2, cov2):
    cov = inv(inv(cov1) + inv(cov2))
    mean = dot(cov, (solve(cov1, mean1) + solve(cov2, mean2)))
    return mean, cov

################################################################################
# Compute product of normal distributions in inverse parametrisation
def product_of_invnormals(v1, L1, v2, L2):
    return v1+v2, L1+L2

################################################################################
# Create a random (but valid) mean and covariance matrix for a k-dimensional Gaussian
def make_gaussian(k):
    mean = random.randn(k)
    A = random.rand(k, k)
    cov = dot(A.T, A)
    return mean, cov

################################################################################
# Reparametrize a measurement likelihood in terms of the state x. This function
# Converts normals parametrized as
#   N( z ; mean, cov )
# into normals parametrized in the inverse representation in x:
#   N^-1( x ; info_vec, info_mat )
# where the measurement z is connected to the state x via a prediction function z=f(x)
#
# The likelihood is p(z | x) = Normal(z ; mean, cov)
#    where mean = prediction and cov = inv(sensor_info)
# The parameters are as follows.
#  - x is a state about which the prediction function has been linearized
#  - prediction is the predicted measurement for the state x -- i.e. the mean of P(z|x).
#  - J is the jacobian of the prediction function (df/dx), evaluated at x
#  - measurement is the actual observation
#  - sensor_info is the inverse of the covariance of the sensor model
#    (i.e. it is the information matrix for z)
#
# The function returns (info_m, info_v), which are the parameters for
# the distribution p(z | x) = Normal(x ; mean, cov)
# (Notice that the above is a normal where the variable is x wheras)
# where:
#   mean = inv(posterior_info_m) * posterior_info_v
#   cov  = inv(posterior_info_m)
def reparametrize_likelihood(x,
                             prediction,
                             Jprediction,
                             measurement,
                             sensor_info):
    residual = measurement - prediction + dot(Jprediction, x)
    # Compute information vector and information matrix for the Gaussian
    v = dots(Jprediction.T, sensor_info, residual)
    L = dots(Jprediction.T, sensor_info, Jprediction)
    return v, L

################################################################################
# Update a belief state given a new observation. The prior belief state is
#   P(x) = N^-1( x ; info_vec, invo_mat )
#
# The measurement is related to the state by a prediction
# function. The most likely measurement (i.e. "prediction") in any
# state x is given by: 
#   z = f(x)
# for some arbiatrary function f. The posterior is:
#   P(x | z) = N^-1( x ; posterior_info_vec , posterior_info_mat )
#
# where posterior_info_vec and posterior_info_mat are the values that
# this function returns.
#
# To do the update, the prediction function must be linearized at some
# point x0. This can theoretically be *any* state, so long as:
#  - the parameter "prediction" is the prediction *at this state*
#  - the parameter "Jprediction" is the jacobian of the prediction func *at this state*
#
# However, the choice of x0 is probably to the quality of the system
# overall, because it can lead to a better/worse approximation to the
# true posterior (we are linearizing, after all). It seems to me that
# it would be good to set this state to some state x0 such that:
#  - the likelihood for the actual observation at x0 is large, i.e. p(z | x0) >> 0
#  - x0 is close to the mean of the prior distribution, so that the
#    linearization is valid around the prior.
#  - perhaps the ideal would be to linearize about the mean of the
#    posterior distribution (yes!)
#  - this could also be chosen as the current best estimate for the
#    state (i.e. taking into account all the in-window data): this
#    would be especially neat since we already have all those values
#    handy following gradient descent!
#
# The parameters are:
#  - prior_v: the information vector for the prior distribution
#  - prior_L: the information matrix for the prior distribution
#  - x0: the state about which to linearize the prediction function (see notes above)
#  - prediction: the predicted measurement at x0
#  - Jprediction: the jacobian of the perdiction function at x0
#  - measurement: the actual observation
#  - sensor_info_mat: the uncertainty of the measurement, expressed as an information matrix
def compute_posterior(prior_v,
                      prior_L,
                      x0,
                      prediction,
                      Jprediction,
                      measurement,
                      sensor_info):
    likelihood_v, likelihood_L = reparametrize_likelihood(x0,
                                                          prediction,
                                                          Jprediction,
                                                          measurement,
                                                          sensor_info)
    posterior_v = prior_v + likelihood_v
    posterior_L = prior_L + likelihood_L
    return posterior_v, posterior_L
    

################################################################################
class BeliefTest(NumpyTestCase):
    def setUp(self):
        # We really need the state to be overconstrained by the
        # measurements in order to do the numeric tests below. To
        # avoid the potential confusion of having size(measurement) ==
        # size(state) we choose a 2d state and 3d measurement

        self.predict = lambda x: array([ x[0]**2, 
                                         x[0]**2 + x[1]**2,
                                         3.*x[1]
                                         ])
        self.Jpredict = lambda x: array([[ 2.*x[0],  0.      ],
                                         [ 2.*x[0],  2.*x[1] ],
                                         [ 0.     ,  3.      ]])

    def test_jacobians(self):
        x0 = array([ 2., 0. ])
        self.assertJacobian(self.predict, self.Jpredict, x0)

    def test_logdet(self):
        mean, cov = make_gaussian(10)
        self.assertAlmostEqual(log(det(cov)),
                               log_det_covariance(cov))

        v,L = normal_to_invnormal(mean, cov)
        self.assertAlmostEqual(log(det(inv(L))),
                               log_det_inv_infomat(L))

    def test_normal_distributions(self):
        x = array([1., 3.5])

        mean = array([2., 3.])
        cov = array([[4., 2.],
                     [2., 3.]])

        v = dot(inv(cov),mean)
        L = inv(cov)

        self.assertAlmostEqual(evaluate_normal(x, mean, cov),
                               evaluate_invnormal(x, v, L))

        f1 = lambda x: evaluate_normal(x, mean, cov)
        self.assertAlmostEqual(integrate2d(f1, mean-[10,10], mean+[10,10]), 1., places=5)

        f2 = lambda x: evaluate_invnormal(x, v, L)
        self.assertAlmostEqual(integrate2d(f2, mean-[10,10], mean+[10,10]), 1., places=5)

        self.assertFunctionsEqual(f1, f2, near=x, radius=5, nsamples=100)

    def test_product(self):
        mean1,cov1 = make_gaussian(5)
        mean2,cov2 = make_gaussian(5)
        
        mean,cov = product_of_normals(mean1, cov1, mean2, cov2)

        v1,L1 = normal_to_invnormal(mean1, cov1)
        v2,L2 = normal_to_invnormal(mean2, cov2)

        v,L = product_of_invnormals(v1, L1, v2, L2)
        mean_,cov_ = invnormal_to_normal(v,L)

        self.assertArrayEqual(mean, mean_)
        self.assertArrayEqual(cov, cov_)
        

    def test_marginalize(self):
        # Setup a Gaussian
        random.seed(245970)
        mean,cov = make_gaussian(5)  # sample a random mean and covariance matrix

        # Pick parameters to keep
        params_to_keep = (0,2,4)

        # Marginalize in ordinary parametrisation
        marg_mean, marg_cov = marginalize_normal(mean, cov, params_to_keep)
        self.assertShape(marg_mean, (3,))
        self.assertShape(marg_cov, (3,3))

        # Marginalize in inverse parametrisation
        v,L = normal_to_invnormal(mean, cov)
        marg_v, marg_L = marginalize_invnormal(v, L, params_to_keep)
        marg_mean2, marg_cov2 = invnormal_to_normal(marg_v, marg_L)
        self.assertShape(marg_mean2, (3,))
        self.assertShape(marg_cov2, (3,3))
        
        # Check that they wind up the same
        self.assertArrayEqual(marg_mean, marg_mean2)
        self.assertArrayEqual(marg_cov, marg_cov2)

    def test_reparam_manual(self):
        x0 = array([ 5., -4. ])
        A = arange(6).reshape((3,2))
        prediction = np.dot(A, x0)
        b = np.zeros(3)
        sensor_cov = eye(3)

        measurement1 = prediction + array([.2,  -1.,  .45])
        measurement2 = prediction + array([.2,  -1.5, .45])
        measurement3 = prediction + array([.25, -1.,   0.])

        z_mean = prediction
        z_cov = sensor_cov
        Pz1 = evaluate_normal(measurement1, z_mean, z_cov)
        Pz2 = evaluate_normal(measurement2, z_mean, z_cov)
        Pz3 = evaluate_normal(measurement3, z_mean, z_cov)

        # alt param
        L = dots(A.T, inv(sensor_cov), A)
        Px1 = evaluate_invnormal(x0, dots(A.T, solve(sensor_cov, measurement1)), L)
        Px2 = evaluate_invnormal(x0, dots(A.T, solve(sensor_cov, measurement2)), L)
        Px3 = evaluate_invnormal(x0, dots(A.T, solve(sensor_cov, measurement3)), L)

        print Px1/Pz1
        print Px2/Pz2
        print Px3/Pz3

        #self.assertAlmostEqual(Pmsm, Pmsm2)

    def test_reparam_square(self):
        x0 = array([ 6., -4. ])
        A = arange(4).reshape((2,2))
        prediction = np.dot(A, x0)
        measurement1 = prediction  + array([ .2,  .45 ])
        measurement2 = prediction + array([ .2,   0. ])
        measurement3 = prediction + array([ .1, -.25 ])
        b = np.zeros(2)
        sensor_cov = 2.*ones((2,2)) + eye(2)*3

        z_mean = prediction
        z_cov = sensor_cov
        Pz1 = evaluate_normal(measurement1, z_mean, z_cov)
        Pz2 = evaluate_normal(measurement2, z_mean, z_cov)
        Pz3 = evaluate_normal(measurement3, z_mean, z_cov)

        # alt param
        #x_mean = 
        #x_cov = inv(dots(A.T, inv(sensor_cov), A))
        #Px1 = evaluate_normal(x0, solve(A, measurement1), x_cov)
        #Px2 = evaluate_normal(x0, solve(A, measurement2), x_cov)
        #Px3 = evaluate_normal(x0, solve(A, measurement3), x_cov)

        L = dots(A.T, inv(sensor_cov), A)
        PPx1 = evaluate_invnormal(x0, dot(A.T, solve(sensor_cov, measurement1)), L)
        PPx2 = evaluate_invnormal(x0, dot(A.T, solve(sensor_cov, measurement2)), L)
        PPx3 = evaluate_invnormal(x0, dot(A.T, solve(sensor_cov, measurement3)), L)

        print Pz1, PPx1
        print Pz2, PPx2
        print Pz3, PPx3

        print PPx1/Pz1
        print PPx2/Pz2
        print PPx3/Pz3
        #self.assertAlmostEqual(Pmsm, Pmsm2)

    def test_reparam_nonsquare(self):
        x0 = array([ 6., -4. ])
        A = arange(6).reshape((3,2))
        prediction = np.dot(A, x0)
        measurement1 = prediction  + array([ .2,  .45,  0. ])
        measurement2 = prediction + array([ .2,    0.,  0. ])
        measurement3 = prediction + array([ .1,  -.25,  0. ])
        b = np.zeros(3)
        sensor_cov = 2.*ones((3,3)) + eye(3)*3

        z_mean = prediction
        z_cov = sensor_cov
        Pz1 = evaluate_normal(measurement1, z_mean, z_cov)
        Pz2 = evaluate_normal(measurement2, z_mean, z_cov)
        Pz3 = evaluate_normal(measurement3, z_mean, z_cov)

        # alt param
        #x_mean = 
        #x_cov = inv(dots(A.T, inv(sensor_cov), A))
        #Px1 = evaluate_normal(x0, solve(A, measurement1), x_cov)
        #Px2 = evaluate_normal(x0, solve(A, measurement2), x_cov)
        #Px3 = evaluate_normal(x0, solve(A, measurement3), x_cov)

        L = dots(A.T, inv(sensor_cov).T, A)
        print L
        PPx1 = evaluate_invnormal(x0, dot(A.T, solve(sensor_cov.T, measurement1)), L)
        PPx2 = evaluate_invnormal(x0, dot(A.T, solve(sensor_cov.T, measurement2)), L)
        PPx3 = evaluate_invnormal(x0, dot(A.T, solve(sensor_cov.T, measurement3)), L)

        z= measurement1
        print dots(z-dot(A,x0), inv(sensor_cov), z-dot(A,x0))
        print 

        print Pz1, PPx1
        print Pz2, PPx2
        print Pz3, PPx3

        print PPx1/Pz1
        print PPx2/Pz2
        print PPx3/Pz3
        #self.assertAlmostEqual(Pmsm, Pmsm2)

    def test_reparametrization(self):
        x0 = array([ 2., -.5 ])

        sensor_cov = diag([ .2, .15, .8 ])
        sensor_infom = inv(sensor_cov)

        prediction = self.predict(x0)
        J = self.Jpredict(x0)
        msm = prediction + [ .01, -.025, .22 ]

        # Likelihood P(z | x), parametrized as a normal in z
        f_z_exact = lambda x: evaluate_normal(msm, self.predict(x), sensor_cov)

        # The same likelihood, linearized at x0. They should be
        # identical at x0 Note the distinction between x and x0. The
        # former is the variable, the latter is the linearization
        # point.
        b = prediction - dot(J, x0)
        f_z_linearized = lambda x: evaluate_normal(msm, dot(J,x)+b, sensor_cov)
        self.assertAlmostEqual(f_z_exact(x0), f_z_linearized(x0))

        # Same likelihood P(z | x), this time parametrized as an inverse-normal in x
        v,L = reparametrize_likelihood(x0, prediction, J, msm, sensor_infom)
        f_x = lambda x: evaluate_invnormal(x, v, L)
        self.assertAlmostEqual(f_x(x0), f_z_linearized(x0))

        xx = array([ 2.,   -1.5 ])
        print 'Jacobian:',
        print J
        print 'v:'
        print v
        print 'L:'
        print L
        print 'JTJ:',
        print dot(J.T,J)
        print 'f_x(xx):', f_x(xx)
        print 'f_z_linearized(xx):', f_z_linearized(xx)
        print 'f_x(x0):', f_x(x0)
        print 'f_z_linearized(x0):', f_z_linearized(x0)

        vv, LL = normal_to_invnormal( dot(J,xx)+b, sensor_cov )
        print 'vv:'
        print vv
        print 'LL:'
        print LL

        f_z_ff = evaluate_invnormal(msm, vv, LL)
        print 'f_z_linearized - computed in inverse repr:',f_z_ff
        

        # Check that the functions are identical
        #self.assertFunctionsEqual(f_x, f_z_linearized, near=x0, radius=3., nsamples=100)


    def test_posterior(self):
        prior_mean = array([ 2., -.5 ])
        prior_cov  = array([[ 5.,   1. ],
                            [ 1.,   4. ]])
        

        # This is the state we linearize about:
        x0 = prior_mean
        
        # Sensor data
        sensor_cov = diag([ .2, .15, .8 ])
        sensor_infom = inv(sensor_cov)

        prediction = self.predict(x0)
        msm = prediction  + [ .01, -.025, .22 ]
        J = self.Jpredict(x0)

        # Likelihood P(z | x), parametrized as a normal in z
        f_z_exact = lambda x: evaluate_normal(msm, self.predict(x), sensor_cov)

        # The same likelihood, linearized at x0. They should be identical at x0
        b = prediction - dot(J, x0)
        f_z_linearized = lambda x: evaluate_normal(msm, dot(J,x)+b, sensor_cov)
        self.assertAlmostEqual(f_z_exact(x0), f_z_linearized(x0))

        

if __name__ == '__main__':
    import unittest
    #seterr(all='raise')
    unittest.main()

        
