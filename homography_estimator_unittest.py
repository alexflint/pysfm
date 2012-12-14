from numpy import *
import unittest

from homography_estimator import *
from numpy_test import NumpyTestCase

################################################################################
def setup_test_problem():
    NUM_CORRESPONDENCES = 12
    MEASUREMENT_NOISE = .2

    # Setup homography
    H = array([[ 1.,   0.,  2.  ],
               [ 0., -.5,  -1.5 ],
               [ .2,  .1,   1.2 ]])

    # Sample points
    random.seed(123)
    xs0 = random.randn(NUM_CORRESPONDENCES,2)
    xs1 = prdot(H, xs0)
    xs0 += random.randn(*xs0.shape) * MEASUREMENT_NOISE
    xs1 += random.randn(*xs1.shape) * MEASUREMENT_NOISE

    # Perturb H so that we do not evaluate jacobians at the minima
    H += random.randn(*H.shape) * .05

    return H, xs0, xs1

################################################################################
# Test jacobians of the sampson error
class HomographyEstimatorTest(NumpyTestCase):
    def setUp(self):
        self.H, self.xs0, self.xs1 = setup_test_problem()

    def test_algebraic_error_jacobian(self):
        x0,x1 = self.xs0[0], self.xs1[1]
        self.assertJacobian(lambda X: algebraic_error(self.H, X[0:2], X[2:4]),
                            algebraic_error_jacobian(self.H, x0, x1),
                            hstack((x0,x1)))

    def test_transfer_error_jacobian(self):
        x0,x1 = self.xs0[0], self.xs1[1]
        self.assertJacobian(lambda h: transfer_error(h.reshape((3,3)), x0, x1),
                            transfer_error_jacobian(self.H, x0, x1),
                            self.H.flatten())

    def test_transfer_error_sl3_jacobian(self):
        x0,x1 = self.xs0[0], self.xs1[1]
        self.assertJacobian(lambda w: transfer_error(dot(self.H, lie.SL3_exp(w)), x0, x1),
                            transfer_error_sl3_jacobian(self.H, x0, x1),
                            zeros(8))

    #def test_symtransfer_error_jacobian(self):
    #    x0,x1 = self.xs0[0], self.xs1[1]
    #    self.assertJacobian(lambda h: symtransfer_error(h.reshape((3,3)), x0, x1),
    #                        symtransfer_error_jacobian(self.H, x0, x1),
    #                        self.H.flatten())

    def test_sampson_error_jacobian(self):
        x0,x1 = self.xs0[0], self.xs1[1]
        self.assertJacobian(lambda h: sampson_error(h.reshape((3,3)), x0, x1),
                            sampson_error_jacobian(self.H, x0, x1),
                            self.H.flatten())

    def test_sampson_error_sl3_jacobian(self):
        x0,x1 = self.xs0[0], self.xs1[1]
        self.assertJacobian(lambda w: sampson_error(dot(self.H, lie.SL3_exp(w)), x0, x1),
                            sampson_error_sl3_jacobian(self.H, x0, x1),
                            zeros(8))

if __name__ == '__main__':
    seterr(all='raise')
    unittest.main()
