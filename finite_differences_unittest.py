from numpy import *
import unittest

from numpy_test import NumpyTestCase
from finite_differences import *

################################################################################
class FiniteDifferencesTest(NumpyTestCase):
    def setUp(self):
        # A simple 1x2 function
        self.f0 = lambda x:array([ 
                x[0]*x[0] + x[1]/x[0],
                ])

        # A simple 2x2 function
        self.f1 = lambda x:array([ 
                x[0]*x[0] + x[1],
                4*x[1]/x[0]
                ])

    def test_numeric_jacobian(self):
        J = array([[4., 1.,],
                   [1., 2. ]])
        self.assertArrayEqual(numeric_jacobian(self.f1, [2., -1]), J)

    def test_numeric_hessian(self):
        H = lambda x,y: array([[[ 2.+2.*y/(x**3) , -1./(x*x) ],
                                [ -1./(x*x)      , 0         ]]])
        x0 = array([3., -.5])
        H0 = H(*x0)
        self.assertArrayEqual(numeric_hessian(self.f0, x0), H0)

################################################################################
if __name__ == '__main__':
    unittest.main()
