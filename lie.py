import numpy as np
import scipy.linalg as la
import finite_differences

G1 = np.array([[ 0., 1., 0. ],
               [ -1., 0., 0. ],
               [ 0., 0., 0. ]])

G2 = np.array([[ 0., 0., 1. ],
               [ 0., 0., 0. ],
               [ -1., 0., 0. ]])

G3 = np.array([[ 0., 0., 0. ],
               [ 0., 0., -1. ],
               [ 0., -1., 0. ]])

Gs = [ G1, G2, G3 ]

class SO3:
    @classmethod
    def exp(cls, m):
        m = np.asarray(m)
        assert np.shape(m) == (3,)
        return la.expm(m[0]*G1 + m[1]*G2 + m[2]*G3)

    @classmethod
    def generator_field(cls, i, x):
        x = np.asarray(x)
        assert np.shape(x) == (3,)
        assert i >= 0 and i < 3
        return np.dot(Gs[i], x)

    # Compute jacobian of exp(m)*x with respect to m. Jacobian is
    # always evaluated at the origin.
    @classmethod
    def J_expm_x(cls, x):
        J = np.empty((3,3))
        for i in range(3):
            J[:,i] = SO3.generator_field(i, x)
        return J




def f(m, x):
    return np.dot(SO3.exp(m), x)

def test():
    x0 = np.array([1., 4., -2.])
    m0 = np.zeros(3)
    
    J = SO3.J_expm_x(x0)
    ff = lambda m: f(m, x0)

    finite_differences.check_jacobian(ff, J, m0)
