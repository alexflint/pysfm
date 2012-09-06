from numpy import *
import numpy as np
import itertools
import unittest
import StringIO

import finite_differences

############################################################################
def spy(A, t=1e-8, stream=None):
    if stream is None:
        import sys
        stream = sys.stdout

    A = asarray(A)
    assert A.ndim <= 2

    if A.ndim == 0:
        A = array([[A]])
    elif A.ndim == 1:
        A = A[newaxis,:]

    for row in A:
        stream.write('[ %s ]\n' % ''.join(take([' ','x'], np.abs(row)>t)))


############################################################################
# Class to represent errors in the jacobian
class ArrayEqualAssertionError(AssertionError):
    def __init__(self, A, B):
        self.A = asarray(A)
        self.B = asarray(B)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        abserr = np.abs(self.A - self.B)

        sio = StringIO.StringIO()
        sio.write('\n')
        if self.B.size < 100:
            sio.write('A:\n')
            sio.write(str(self.A)+'\n')
            sio.write('B:\n')
            sio.write(str(self.B)+'\n')
            sio.write('Abs difference:\n')
            sio.write(str(abserr)+'\n')

        if self.A.size > 10 and self.A.ndim <= 2:
            sio.write('A (sparsity pattern):\n')
            spy(self.A, 1e-5, stream=sio)
            sio.write('B (sparsity pattern):\n')
            spy(self.B, 1e-5, stream=sio)
            sio.write('Abs difference (sparsity pattern):\n')
            spy(abserr, 1e-5, stream=sio)

        i = unravel_index(argmax(abserr), abserr.shape)
        sio.write('Max error: %f\n' % np.max(abserr))
        sio.write('  (analytic=%f vs numeric=%f at %s)\n' % \
                      (self.A[i], self.B[i], tuple(i)))
    
        return sio.getvalue()

############################################################################
# Class to represent errors in the jacobian
class JacobianAssertionError(AssertionError):
    def __init__(self, J_numeric, J_analytic):
        self.J_numeric = J_numeric
        self.J_analytic = J_analytic

    def __str__(self):
        return repr(self)

    def __repr__(self):
        J_abserr = np.abs(self.J_numeric - self.J_analytic)

        sio = StringIO.StringIO()
        sio.write('\n')
        if self.J_numeric.size < 100:
            sio.write('Numeric jacobian:\n')
            sio.write(str(self.J_numeric)+'\n')
            sio.write('Analytic jacobian:\n')
            sio.write(str(self.J_analytic)+'\n')
            sio.write('Residual of jacobian:\n')
            sio.write(str(J_abserr)+'\n')

        if self.J_numeric.size > 10:
            sio.write('Numeric Jacobian (sparsity pattern):\n')
            spy(self.J_numeric, 1e-5, stream=sio)
            sio.write('Analytic Jacobian (sparsity pattern):\n')
            spy(self.J_analytic, 1e-5, stream=sio)
            sio.write('Residual of Jacobian (sparsity pattern):\n')
            spy(J_abserr, 1e-5, stream=sio)

        i = unravel_index(argmax(J_abserr), J_abserr.shape)
        sio.write('Max error in Jacobian: %f\n' % np.max(J_abserr))
        sio.write('  (analytic=%f vs numeric=%f at %s)\n' % \
                      (self.J_analytic[i], self.J_numeric[i], tuple(i)))
    
        return sio.getvalue()

############################################################################
# Helpers for numpy testing

class NumpyTestCase(unittest.TestCase):
    def assertShape(self, arr, expected_shape):
        self.assertEqual(shape(arr), expected_shape)


    # Check that two arrays are equal (up to machine precision)
    def assertArrayEqual(self, A, B):
        A = asarray(A)
        B = asarray(B)
        self.assertEqual(A.shape, B.shape)
        err = np.sum(square(A-B))
        if err > 1e-7:
            raise ArrayEqualAssertionError(A, B)


    # Check that two functions are equal (up to machine precision)
    def assertFunctionsEqual(self, f1, f2, near, radius, nsamples, tol=1e-5):
        near = atleast_1d(asarray(near))
        lo = near-radius
        hi = near+radius
        n = ceil(nsamples ** (1. / len(near)))
        samples = [ linspace(xa,xb,n) for xa,xb in zip(lo,hi) ]
        for i,x in enumerate(itertools.product(*samples)):
            f1x = f1(x)
            f2x = f2(x)
            err = abs(f1x-f2x)
            if err > tol:
                raise AssertionError, 'At x=%s:\nf1(x) = %f\nf2(x) = %f\nerr = %f' % \
                    (str(x), f1x, f2x, err)
            if i == nsamples-1:
                break


    # Check that the Jacobian of f (computed numerically) equals Jf
    def assertJacobian(self, f, Jf, x0, tol=1e-5, h=1e-8):
        x0 = asarray(x0)

        # if Jf is a function then just evaluate it once
        if callable(Jf):
            J_analytic = atleast_2d(Jf(x0))
        else:
            J_analytic = atleast_2d(Jf)

        # Compute numeric jacobian
        J_numeric = finite_differences.numeric_jacobian(f, x0, h)

        # Check shape
        self.assertEqual(J_analytic.shape, J_numeric.shape)

        # Compute error
        J_abserr = np.abs(J_analytic - J_numeric)
        mask = np.abs(J_numeric) > 1.
        J_abserr[mask] /= np.abs(J_numeric[mask])

        # Check error
        maxerr = np.max(J_abserr)  # threshold should be independent of matrix size
        if maxerr > tol:
            raise JacobianAssertionError(J_numeric, J_analytic)


