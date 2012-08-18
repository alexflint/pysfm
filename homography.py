import numpy as np
import matplotlib.pyplot as plt

from algebra import *

# Build a homography from w
def H(w):
    return np.array([[ w[0], 0.,   w[2] ],
                     [ 0.,   w[1], w[3] ],
                     [ 0.,   0.,   1.   ]])

wtrue = np.array([2.5, 1.5, -1., 0.])
Htrue = H(wtrue)

X = np.array([[0., 0.],
              [3., 2.],
              [0., 1.]])

Y = prdot(Htrue, X)

def f(w):
    assert np.shape(w) == (4,), 'w had shape: '+str(np.shape(w))
    return (prdot(H(w), X) - Y).flatten()

def c(w):
    fw = f(w)
    return np.dot(fw, fw)

def Jf(w):
    npts = X.shape[0]
    J = np.empty((2*npts, 4))
    for i in range(npts):
        J[i*2]   = [ X[i,0], 0., 1., 0. ]
        J[i*2+1] = [ 0., X[i,1], 0., 1. ]
    return J     

def check_jacobians(w0):
    w0 = np.asarray(w0)

    # compute gradients numerically
    J_numeric = numeric_jacobian(f, w0, 1e-8)
    grad_numeric = numeric_jacobian(c, w0, 1e-8)

    # check analytic gradients
    f0 = f(w0)
    J_ana = Jf(w0)

    print f0.shape
    print J_ana.shape

    grad_ana = 2. * np.dot(f0, J_ana)
    
    J_err = np.abs(J_ana - J_numeric)
    grad_err = np.abs(grad_ana - grad_numeric)

    print 'Numeric gradient:'
    print grad_numeric
    print 'Analytic gradient:'
    print grad_ana
    print 'Residual of gradient:'
    print grad_err
    print 'Error in gradient:'
    print np.linalg.norm(grad_err)

    print 'Numeric jacobian:'
    print J_numeric
    print 'Analytic jacobian:'
    print J_ana
    print 'Residual of jacobian:'
    print J_err
    print 'Error in jacobian:'
    print np.linalg.norm(J_err)


def optimize_first_order():
    kMinStepSize = 1e-8
    kMaxSteps = 10

    w0 = np.zeros(4)
    w_path = [ w0 ]

    c_prev = None

    nsteps = 0
    stepsize = 1.
    w_cur = w0;
    while nsteps < kMaxSteps:
        f_cur = f(w_cur)
        c_cur = np.dot(f_cur, f_cur)
        Jf_cur = Jf(w_cur)
        grad_cur = 2. * np.dot(f_cur, Jf_cur)

        grad_numeric = numeric_jacobian(c, w_cur, 1e-8)

        print 'Cost: %f' % c_cur
        print '  Residuals: '+str(f_cur)
        print '  Gradient: '+str(grad_cur)
        print '  Numeric gradient: '+str(grad_numeric)
        print '  Step size: '+str(stepsize)
        
        while stepsize > kMinStepSize:
            w_next = w_cur - grad_cur * stepsize;
            c_next = c(w_next)
            if (c_next < c_cur):
                w_cur = w_next
                w_path.append(w_cur)
                stepsize *= 1.1
                break
            else:
                stepsize *= .5

        nsteps += 1
            
    # print and return
    plt.clf()
    plt.xlim(-5, 10)
    plt.ylim(-5, 5)
    plt.plot(Y[:,0], Y[:,1], 'og')

    colors = 'rgbcmyk'
    C = colors[:len(X)]

    for x,color in zip(X,C):
        x_path = np.array([ prdot(H(w), x) for w in w_path ])
        plt.plot(x_path[:,0], x_path[:,1], 'x-'+color)

    plt.savefig('first_order_path.pdf')

def solve_normal_equations(J, r):
    A = np.dot(J.T, J)
    b = np.dot(J.T, r)
    return np.linalg.solve(A, b)

def optimize_second_order():
    kMinStepSize = 1e-8
    kMaxSteps = 10

    w0 = np.zeros(4)
    w_path = [ w0 ]

    c_prev = None

    nsteps = 0
    stepsize = 1.
    w_cur = w0;
    while nsteps < kMaxSteps:
        f_cur = f(w_cur)
        c_cur = np.dot(f_cur, f_cur)
        Jf_cur = Jf(w_cur)
        descent_direction = solve_normal_equations(Jf_cur, f_cur)

        print 'Cost: %f' % c_cur
        print '  Parameters: '+str(w_cur)
        print '  Residuals: '+str(f_cur)
        print '  Descent direction: '+str(descent_direction)
        print '  Step size: '+str(stepsize)
        
        while stepsize > kMinStepSize:
            w_next = w_cur - descent_direction * stepsize;
            c_next = c(w_next)
            if (c_next < c_cur):
                w_cur = w_next
                w_path.append(w_cur)
                stepsize *= 1.1
                break
            else:
                stepsize *= .5

        nsteps += 1
            
    # print and return
    plt.clf()
    plt.xlim(-5, 10)
    plt.ylim(-5, 5)
    plt.plot(Y[:,0], Y[:,1], 'og')

    colors = 'rgbcmyk'
    C = colors[:len(X)]

    for x,color in zip(X,C):
        x_path = np.array([ prdot(H(w), x) for w in w_path ])
        plt.plot(x_path[:,0], x_path[:,1], 'x-'+color)

    plt.savefig('second_order_path.pdf')

if __name__ == '__main__':
    check_jacobians(np.zeros(4))
    check_jacobians([ 1., 2., -0.5, 3.11 ])
    #optimize_first_order()
    optimize_second_order()
