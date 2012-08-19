import numpy as np
from algebra import *

# Represents an error model in which the cost of an error is the
# squared Euclidean norm of the error vector.
class GaussianModel(object):
    def __init__(self, cov=1.):
        if np.isscalar(cov):
            self.cov = cov * np.eye(2)
        elif np.shape(cov) == (2,):
            self.cov = np.diag(cov)
        else:
            assert np.shape(cov) == (2,2)
            self.cov = np.asarray(cov)
        self.covinv = np.linalg.inv(self.cov)
        self.L = np.linalg.cholesky(self.covinv)

    def cost_from_error(self, x):
        assert np.shape(x) == (2,)
        return dots(x, self.covinv, x)

    def residual_from_error(self, x):
        assert np.shape(x) == (2,)
        return np.dot(self.L, x)

    def Jresidual_from_error(self, x):
        assert np.shape(x) == (2,)
        return self.L

# Represents an error model in which errors are penalized according to
# a Cauchy robustifier.
class CauchyModel(object):
    # Within this window, the Cauchy function is approximated as linear
    LinearWindowAboutZero = 1e-5

    def __init__(self, sigma):
        self.sigma = sigma
        self.sigmasqr = sigma * sigma

    def cost_from_error(self, x):
        return np.log(1. + ssq(x)/self.sigmasqr)

    def residual_from_error(self, x):
        assert np.shape(x) == (2,)
        x = np.asarray(x)
        r = np.linalg.norm(x)
        if r < CauchyModel.LinearWindowAboutZero:
            return x / self.sigma

        s = self.sigma
        e = np.sqrt(np.log(1. + r*r / (s*s)))
        return x * e / r

    def Jresidual_from_error(self, x):
        assert np.shape(x) == (2,)
        x = np.asarray(x)
        r = np.linalg.norm(x)
        if r < CauchyModel.LinearWindowAboutZero:
            return np.eye(2) / self.sigma

        xx = np.outer(x,x)
        e = np.sqrt(np.log(1. + r*r / self.sigmasqr))
        I = np.eye(2)
        return xx / (r*e*(r*r + self.sigmasqr)) + (r*I - xx/r) * e/(r*r)


################################################################################
# Validation of sensor models
def validate(sensor_model):
    # Check residuals
    error = np.asarray([1., 2.])
    cost = sensor_model.cost_from_error(error)
    residual = sensor_model.residual_from_error(error)
    assert np.abs(cost - np.dot(residual,residual)) < 1e-8, \
        'cost was not equalt to residual.T * residual'

    # Check cost at zero
    cost_at_0 = sensor_model.cost_from_error([0,0])
    assert np.isscalar(cost_at_0)
    assert abs(cost_at_0) < 1e-8, 'Cost at 0 must be 0'

    # Check jacobians
    J = sensor_model.Jresidual_from_error(error)
    assert J.shape == (2,2), 'shape was %s' % str(J.shape)

    import finite_differences
    err,J_err = finite_differences.check_jacobian(sensor_model.residual_from_error,
                                                  J,
                                                  error)
    assert abs(err) < 1e-5, 'Jacobian seems to be incorrect at '+str(error)

    print 'SENSOR MODEL PASSED TESTS'


################################################################################
# Unit tests
def run_tests():
    print 'Validating Gaussian sensor model...'
    validate(GaussianModel([2., 3.]))

    print 'Validating Cauchy sensor model...'
    validate(CauchyModel(2.))
