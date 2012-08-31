from numpy import *
from numpy.linalg import *
import numpy as np

from numpy_test import NumpyTestCase
from lie import SO3
from geometry import *
from algebra import *

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

x0 = xs0[0] + (.5,   0.,  1.)
x1 = xs1[0] + (-1., .8,   1.)


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
        #f = lambda v: residual(K, R, v[:3], v[3:], x0, x1)
        self.assertJacobian(fR, Jresidual(K, R, t, x0, x1)[newaxis,:3], zeros(3))
        #self.assertJacobian( f, Jresidual(K, R, t, x0, x1)[newaxis,:], concatenate((zeros(3),t)) )


if __name__ == '__main__':
    import unittest
    np.seterr(all='raise')
    unittest.main()

    test_optimize()
