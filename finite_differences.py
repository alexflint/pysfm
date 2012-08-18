import sys
import numpy as np

def spy(A, t=1e-8):
    A = np.asarray(A)
    assert A.ndim == 2
    for row in A:
        sys.stdout.write('[')
        for x in row:
            if abs(x) > t:
                sys.stdout.write('x')
            else:
                sys.stdout.write(' ')
        sys.stdout.write(']\n')

def axis(i, ndim):
    x = np.zeros(ndim)
    x[i] = 1.
    return x

def numeric_derivative(f, x0, h=1e-8):
    assert np.isscalar(x0), 'numeric_derivative can only deal with one-dimensional inputs'
    fa = f(x0 - h)
    fb = f(x0 + h)
    return (fb - fa) / (2. * h)

def numeric_jacobian(f, x0, h=1e-8):
    f0 = f(x0)
    J_numeric = np.empty((len(f0), len(x0)))
    for i in range(len(x0)):
        f_partial = lambda delta: f(x0 + axis(i, len(x0))*delta)
        J_numeric[:,i] = numeric_derivative(f_partial, 0., h)
    return J_numeric

def check_jacobian(f, Jf, x0):
    # if Jf is a function then just evaluate it once
    if callable(Jf):
        Jf = Jf(x0)

    x0 = np.asarray(x0)
    Jf_numeric = numeric_jacobian(f, x0, 1e-8)

    if Jf.shape != Jf_numeric.shape:
        print 'Error: shape mismatch'
        print '  analytic jacobian was '+str(Jf.shape)
        print '  numeric jacobian was '+str(Jf_numeric.shape)
        return False

    Jf_abserr = np.abs(Jf - Jf_numeric)
    mask = np.abs(Jf_numeric) > 1.
    Jf_abserr[mask] /= np.abs(Jf_numeric[mask])

    abserr = np.max(Jf_abserr)  # threshold should be independent of matrix size
    if abserr < 1e-5:
        print 'JACOBIAN IS CORRECT'
    else:
        if Jf.size < 25:
            print 'Numeric jacobian:'
            print Jf_numeric
            print 'Analytic jacobian:'
            print Jf
            print 'Residual of jacobian:'
            print Jf_abserr
        else:
            print 'Numeric Jacobian (sparsity pattern):'
            spy(Jf_numeric)
            print 'Analytic Jacobian (sparsity pattern):'
            spy(Jf)
            print 'Residual of Jacobian (sparsity pattern):'
            spy(Jf_abserr, 1e-6)

        print 'Max error in Jacobian:'
        print abserr
        i = np.unravel_index(np.argmax(Jf_abserr), Jf_abserr.shape)
        print '  (%f vs %f at %d,%d)' % (Jf[i], Jf_numeric[i], i[0], i[1])
        print 'JACOBIAN IS WRONG'

    return abserr, Jf_abserr
