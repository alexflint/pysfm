import numpy as np

from algebra import *

############################################################################
# Levenberg Marquardt
def apply_lm_damping_inplace(A, damping):
    A = np.asarray(A)
    A[ np.diag_indices(A.shape[0]) ] *= (1. + damping)

def apply_lm_damping(A, damping):
    A = np.asarray(A)
    B = A.copy()
    apply_lm_damping_inplace(B)
    return B

############################################################################
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

def gauss_newton_step(J, r):
    A = np.dot(J.T, J)
    b = np.dot(J.T, r)
    return np.linalg.solve(A, b)

def levenberg_marquardt_step(J, r, damp):
    A = np.dot(J.T, J)
    b = np.dot(J.T, r)
    A[np.diag_indices_from(A)] *= (1. + damp)
    return np.linalg.solve(A, b)

class LevenbergMarquardtOptimizer:
    def __init__(self, f, Jf, init_damp=100.):
        self.f = f
        self.Jf = Jf
        self.ycur = None
        self.Jcur = None
        self.cost = None
        self.damp = init_damp
        self.path = []   # will be populated with the list of evaluations
        self.num_steps = 0
        self.param_mask = None

        self.max_steps = 100
        self.min_damp = 1e-8
        self.max_damp = 1e+8
        self.min_improvement = 1e-8

    def reset(self, x0):
        self.xcur = x0
        self.ycur = self.f(self.xcur)
        self.Jcur = self.Jf(self.xcur)
        self.cost = np.dot(self.ycur, self.ycur)
        self.num_steps = 0
        self.path = [self.xcur]
        if self.param_mask is None:
            self.param_mask = np.array([True]*len(x0))

    def step(self):
        assert self.xcur is not None, 'reset() must be called before step()'
        prevcost = self.cost

        while self.damp < self.max_damp:
            J_masked = self.Jcur[ : , self.param_mask ]
            step = -levenberg_marquardt_step(J_masked, self.ycur, self.damp)
            x_next = self.xcur
            x_next[self.param_mask] += step
            y_next = self.f(x_next)
            c_next = np.dot(y_next, y_next)
            if c_next < self.cost:
                self.xcur = x_next
                self.ycur = y_next
                self.Jcur = self.Jf(self.xcur)
                self.cost = c_next
                self.path.append(self.xcur)
                self.num_steps += 1
                self.damp = max(self.min_damp, self.damp * .1)
                break
            else:
                self.damp *= 10.

        return \
            abs(prevcost - self.cost) < self.min_improvement or \
            self.damp >= self.max_damp

    def optimize(self, x0):
        self.reset(x0)
        for i in range(self.max_steps):
            converged = self.step()
            print 'Step %d: cost= %f, damp=%f' % (i,self.cost,self.damp)
            if converged:
                print 'Converged'
                return True
        print 'Expired without converging'
        return False

class LevenbergMarquardt:
    def __init__(self, init_damp=100.):
        self.init_damp = init_damp
        self.damp = init_damp
        self.num_steps = 0
        self.num_normal_solutions = 0

        self.max_steps = 100
        self.min_damp = 1e-8
        self.max_damp = 1e+8
        self.min_improvement = 1e-8
        self.converged = False

    def reset(self):
        self.damp = init_damp
        self.num_steps = 0
        self.num_normal_solutions = 0

    def next_update(self, r, J):
        self.num_normal_solutions += 1
        self.last_cost = np.dot(r, r)
        return -levenberg_marquardt_step(J, r, self.damp)

    def next_damping_value(self):
        return self.damp

    def accept_update(self, rnew):
        self.num_steps += 1
        self.damp = max(self.min_damp, self.damp * .1)
        updated_cost = np.dot(rnew, rnew)
        self.converged = np.abs(updated_cost - self.last_cost) < self.min_improvement

    def reject_update(self):
        self.damp *= 10.
        self.converged = self.damp > self.max_damp



def foo():            
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
