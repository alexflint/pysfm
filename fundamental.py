from numpy import *
from numpy.linalg import *
import numpy as np

from numpy_test import NumpyTestCase
from lie import SO3
from geometry import *
from algebra import *

SENSOR_NOISE = 0

MAX_STEPS = 50
CONVERGENCE_THRESH = 1e-5   # convergence detected when improvement less than this
         
INIT_DAMPING = .1

# Compute the fundamental matrix K^-T * R * skew(x) * K^-1
def fund(K, R, t):
    return dots(inv(K).T, skew(t), R, inv(K))



def F1x(K, R0, m, t, x):
    R = dot(R0, SO3.exp(m))
    F = fund(K, R, t)
    return dot(F[0], x)

def JF1x_R(K, R, t, x):
    v = dots(R.T, skew(t), inv(K))[:,0]
    JF1_R = dots(inv(K).T, skew(-v))
    return dot(JF1_R.T, x)

def JF1x_t(K, R, t, x):
    v = inv(K)[:,0]
    return -dot(dots(inv(K), R.T, skew(-v)).T, x)

def JF1x(K, R, t, x):
    return np.concatenate((JF1x_R(K, R, t, x),
                           JF1x_t(K, R, t, x)))


def FT1x(K, R0, m, t, x):
    R = dot(R0, SO3.exp(m))
    F = fund(K, R, t)
    return dot(F[:,0], x)

def JFT1x_R(K, R, t, x):
    v = inv(K)[:,0]
    return dot(dots(inv(K).T, skew(t), R, skew(-v)).T, x)

def JFT1x_t(K, R, t, x):
    v = dot(R, inv(K))[:,0]
    return dot(dots(inv(K).T, skew(-v)).T, x)

def JFT1x(K, R, t, x):
    return np.concatenate((JFT1x_R(K, R, t, x),
                           JFT1x_t(K, R, t, x)))





def F2x(K, R0, m, t, x):
    R = dot(R0, SO3.exp(m))
    F = fund(K, R, t)
    return dot(F[1], x)

def JF2x_R(K, R, t, x):
    v = dots(R.T, skew(t), inv(K))[:,1]
    JF1_R = dots(inv(K).T, skew(-v))
    return dot(JF1_R.T, x)

def JF2x_t(K, R, t, x):
    v = inv(K)[:,1]
    return -dot(dots(inv(K), R.T, skew(-v)).T, x)

def JF2x(K, R, t, x):
    return np.concatenate((JF2x_R(K, R, t, x),
                           JF2x_t(K, R, t, x)))




def FT2x(K, R0, m, t, x):
    R = dot(R0, SO3.exp(m))
    F = fund(K, R, t)
    return dot(F[:,1], x)

def JFT2x_R(K, R, t, x):
    v = inv(K)[:,1]
    return dots(dots(inv(K).T, skew(t), R, skew(-v)).T, x)

def JFT2x_t(K, R, t, x):
    v = dot(R, inv(K))[:,1]
    return dot(dots(inv(K).T, skew(-v)).T, x)

def JFT2x(K, R, t, x):
    return np.concatenate((JFT2x_R(K, R, t, x),
                           JFT2x_t(K, R, t, x)))




def xFx(K, R0, m, t, x0, x1):
    R = dot(R0, SO3.exp(m))
    return dots(x1, fund(K, R, t), x0)

def JxFx_R(K, R, t, x0, x1):
    v = dot(inv(K), x0)
    return dots(x1, inv(K).T, skew(t), R, skew(-v))

def JxFx_t(K, R, t, x0, x1):
    v = dots(R, inv(K), x0)
    return dots(x1, inv(K).T, skew(-v))

def JxFx(K, R, t, x0, x1):
    return np.concatenate((JxFx_R(K, R, t, x0, x1),
                           JxFx_t(K, R, t, x0, x1)))





def residual(K, R0, m, t, x0, x1):
    f = xFx(K, R0, m, t, x0, x1)
    g1 = F1x(K, R0, m, t, x0)
    g2 = F2x(K, R0, m, t, x0)
    g3 = FT1x(K, R0, m, t, x1)
    g4 = FT2x(K, R0, m, t, x1)
    return f / sqrt(g1*g1 + g2*g2 + g3*g3 + g4*g4)

def Jresidual(K, R, t, x0, x1):
    m = zeros(3)

    f = xFx(K, R, m, t, x0, x1)
    Jf = JxFx(K, R, t, x0, x1)

    g1 = F1x(K, R, m, t, x0)
    Jg1 = JF1x(K, R, t, x0)

    g2 = F2x(K, R, m, t, x0)
    Jg2 = JF2x(K, R, t, x0)

    g3 = FT1x(K, R, m, t, x1)
    Jg3 = JFT1x(K, R, t, x1)

    g4 = FT2x(K, R, m, t, x1)
    Jg4 = JFT2x(K, R, t, x1)

    d = g1*g1 + g2*g2 + g3*g3 + g4*g4
    Jd = g1*Jg1 + g2*Jg2 + g3*Jg3 + g4*Jg4
    J = Jf/sqrt(d) - f*Jd / (d * sqrt(d))
    return J

def cost(K, R, t, xs0, xs1):
    c = 0.
    for x0,x1 in zip(xs0,xs1):
        c += square(residual(K, R, zeros(3), t, x0, x1))
    return c


K = eye(3)

R0 = eye(3)
t0 = zeros(3)
P0 = hstack((R0, t0[:,newaxis]))

R1 = dots(rotation_xz(-.2), rotation_xy(.1), rotation_xz(1.5))
t1 = np.array([1., .5, -2.])
P1 = hstack((R1, t1[:,newaxis]))

R,t = relative_pose(R0,t0, R1,t1)
F = fund(K, R, t)


random.seed(123)
pts = random.randn(10,3) + array([0., 0., -5.])

xs0 = array([ dot(K, dot(R0, x) + t0) for x in pts ])
xs1 = array([ dot(K, dot(R1, x) + t1) for x in pts ])

xs0 += random.randn(*xs0.shape) * SENSOR_NOISE
xs1 += random.randn(*xs1.shape) * SENSOR_NOISE

x0 = xs0[0] + (.5,   0.,  1.)
x1 = xs1[0] + (-1., .8,   1.)


################################################################################
def optimize_fmatrix(xs0, xs1, R_init, t_init):
    num_steps = 0
    R_cur = R_init.copy()
    t_cur = t_init.copy()
    cost_cur = cost(K, R_cur, t_cur, xs0, xs1)
    converged = False
    damping = INIT_DAMPING
    while num_steps < MAX_STEPS and damping < 1e+8 and not converged:
        print 'Step %d: cost=%f' % (num_steps, cost_cur)

        # Pick a parameter to freeze
        # TODO: do this the proper way instead
        freeze = 3 + argmax(t_cur)
        mask = arange(6) != freeze

        # Compute J^T * J and J^T * r
        JTJ = zeros((5,5))  # 6x6
        JTr = zeros(5)      # 6x1
        for x0,x1 in zip(xs0,xs1):
            # Jacobian for the i-th point:
            ri = residual(K, R_cur, zeros(3), t_cur, x0, x1)
            Ji = Jresidual(K, R_cur, t_cur, x0, x1)[ mask ]
            # Add to full jacobian
            JTJ += dot(Ji.T, Ji)    # 6x2 * 2x6 -> 6x6
            JTr += dot(Ji.T, ri)    # 6x2 * 2x1 -> 6x1

        # Pick a step length
        while damping < 1e+8 and not converged:
            # Check for failure
            if damping > 1e+8:
                print 'FAILED TO CONVERGE'
                break

            # Apply Levenberg-Marquardt damping
            A = JTJ.copy()
            for i in range(A.shape[0]):
                A[i,i] *= (1. + damping)
    
            # Solve normal equations: 5x5
            try:
                update = -solve(A, JTr)
            except LinAlgError:
                # Badly conditioned: increase damping and try again
                damping *= 10.
                continue

            # Take gradient step
            R_next = dot(R_cur, SO3.exp(update[:3]))
            t_next = t_cur.copy()
            t_next[mask[3:]] = t_cur[mask[3:]] + update[3:]

            # Compute new cost
            cost_next = cost(K, R_next, t_next, xs0, xs1)
            print '  cur cost',cost_cur
            print '  next cost',cost_next
            if cost_next < cost_cur:
                # Cost decreased: accept the update
                if cost_cur - cost_next < CONVERGENCE_THRESH or damping < 1e-8:
                    converged = True

                R_cur = R_next
                t_cur = t_next
                cost_cur = cost_next
                damping *= .1
                num_steps += 1
                break

            else:
                # Cost increased: reject the udpate, try another step length
                damping *= 10.
                print '  Update rejected... increasing damping to '+str(damping)

    return R_cur, t_cur



def test_optimize():
    random.seed(73)
    R_init = dot(R, SO3.exp(random.randn(3)*.1))
    t_init = t + random.randn(3)*.1
    R_opt, t_opt = optimize_fmatrix(xs0, xs1, R_init, t_init)
    print 'True:'
    print np.hstack((R, t[:,newaxis]))
    print 'Initial:'
    print np.hstack((R_init, t_init[:,newaxis]))
    print 'Final:'
    print np.hstack((R_opt, t_opt[:,newaxis]))

################################################################################
class FundmentalMatrixTest(NumpyTestCase):
    def setUp(self):
        pass

    def test_JF1x(self):
        fR = lambda m:  F1x(K, R, m, t, x)
        ft = lambda tt: F1x(K, R, zeros(3), tt, x)
        self.assertJacobian(fR, JF1x_R(K, R, t, x)[newaxis,:], zeros(3))
        self.assertJacobian(ft, JF1x_t(K, R, t, x)[newaxis,:], t)

    def test_JFT1x(self):
        fR = lambda m: FT1x(K, R, m, t, x)
        ft = lambda tt: FT1x(K, R, zeros(3), tt, x)
        self.assertJacobian(fR, JFT1x_R(K, R, t, x)[newaxis,:], zeros(3))
        self.assertJacobian(ft, JFT1x_t(K, R, t, x)[newaxis,:], t)

    def test_JF2x(self):
        fR = lambda m:  F2x(K, R, m, t, x)
        ft = lambda tt: F2x(K, R, zeros(3), tt, x)
        self.assertJacobian(fR, JF2x_R(K, R, t, x)[newaxis,:], zeros(3))
        self.assertJacobian(ft, JF2x_t(K, R, t, x)[newaxis,:], t)

    def test_JFT2x(self):
        fR = lambda m: FT2x(K, R, m, t, x)
        ft = lambda tt: FT2x(K, R, zeros(3), tt, x)
        self.assertJacobian(fR, JFT2x_R(K, R, t, x)[newaxis,:], zeros(3))
        self.assertJacobian(ft, JFT2x_t(K, R, t, x)[newaxis,:], t)

    def test_JxFx(self):
        fR = lambda m: xFx(K, R, m, t, x0, x1)
        ft = lambda tt: xFx(K, R, zeros(3), tt, x0, x1)
        self.assertJacobian(fR, JxFx_R(K, R, t, x0, x1)[newaxis,:], zeros(3))
        self.assertJacobian(ft, JxFx_t(K, R, t, x0, x1)[newaxis,:], t)

    def test_Jresidual(self):
        fR = lambda m: residual(K, R, m, t, x0, x1)
        f = lambda v: residual(K, R, v[:3], v[3:], x0, x1)
        self.assertJacobian(fR, Jresidual(K, R, t, x0, x1)[newaxis,:3], zeros(3))
        self.assertJacobian( f, Jresidual(K, R, t, x0, x1)[newaxis,:], concatenate((zeros(3),t)) )


if __name__ == '__main__':
    test_optimize()

    import unittest
    np.seterr(all='raise')
    unittest.main()

