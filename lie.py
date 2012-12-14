import numpy as np
import scipy.linalg as la
from numpy_test import NumpyTestCase
from algebra import skew

Gs = np.array([[[ 0.,  0.,  0. ],
                [ 0.,  0., -1. ],
                [ 0.,  1.,  0. ]],

               [[ 0.,  0.,  1. ],
                [ 0.,  0.,  0. ],
                [ -1., 0.,  0. ]],

               [[ 0., -1.,  0. ],
                [ 1.,  0.,  0. ],
                [ 0.,  0.,  0. ]]])

################################################################################
class SO3(object):
    # Compute the mapping from so(3) to SO(3)
    @classmethod
    def exp(cls, m):
        m = np.asarray(m)
        assert np.shape(m) == (3,), 'shape was '+str(np.shape(m))

        t = np.linalg.norm(m)
        if t < 1e-8:
            return np.eye(3)   # exp(0) = I

        skewm = skew(m)
        A = np.sin(t)/t
        B = (1. - np.cos(t)) / (t*t)
        I = np.eye(3)
        return I + A * skewm + B * np.dot(skewm, skewm)

    # Compute jacobian of exp(m)*x with respect to m, evaluated at
    # m=[0,0,0]. x is assumed constant with respect to m.
    @classmethod
    def J_expm_x(cls, x):
        return skew(-x)

    # Return the generators times x
    @classmethod
    def generator_field(cls, x):
        return skew(x)

    # Compute the exponential of m*Gs, slow way
    @classmethod
    def exp_slow(cls, m):
        m = np.asarray(m)
        assert np.shape(m) == (3,)
        return la.expm(m[0]*Gs[0] + m[1]*Gs[1] + m[2]*Gs[2])

    # Compute jacobian of exp(m)*x with respect to m, evaluated at m=[0,0,0], slow way
    @classmethod
    def J_expm_x_slow(cls, x):
        J = np.empty((3,3))
        for i in range(3):
            J[:,i] = SO3.generator_field(i, x)
        return J

################################################################################
def make_SL3_basis():
    Gs = np.zeros((8,3,3))
    Gs[0,0,2] =  1.
    Gs[1,1,2] =  1.
    Gs[2,0,1] =  1.
    Gs[3,1,0] =  1.
    Gs[4,0,0] =  1.
    Gs[4,1,1] = -1.
    Gs[5,1,1] = -1.
    Gs[5,2,2] =  1.
    Gs[6,2,0] =  1.
    Gs[7,2,1] =  1.
    return Gs

SL3_BASIS = make_SL3_basis()

def SL3_exp(w):
    import scipy.linalg
    assert np.shape(w) == (8,)
    return scipy.linalg.expm(np.sum( w[i] * SL3_BASIS[i] for i in range(8) ))


################################################################################
class LieTest(NumpyTestCase):
    def test_rodrigues(self):
        m = np.array([1., 3., -1.])
        R1 = SO3.exp(m)
        R2 = SO3.exp_slow(m)
        self.assertArrayEqual(R1, R2)

    def test_generator_field(self):
        m = np.array([1., 3., -1.])
        self.assertArrayEqual(SO3.generator_field(m),
                              m[0]*Gs[0] + m[1]*Gs[1] + m[2]*Gs[2])

    def test_jacobian(self):
        x0 = np.array([1., 4., -2.])
        J = SO3.J_expm_x(x0)
        f = lambda m: np.dot(SO3.exp(m), x0)
        self.assertJacobian(f, J, np.zeros(3))

if __name__ == '__main__':
    import unittest
    unittest.main()
