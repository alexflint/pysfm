from numpy import *
import unittest

from fundamental import *
from numpy_test import NumpyTestCase
import finite_differences

################################################################################
def setup_test_problem():
    noise = .1

    R0 = eye(3)
    t0 = zeros(3)
    P0 = hstack((R0, t0[:,newaxis]))

    R1 = dots(rotation_xz(-.2), rotation_xy(.1), rotation_yz(1.5))
    t1 = np.array([-1., .5, -2.])
    P1 = hstack((R1, t1[:,newaxis]))
    
    R,t = relative_pose(R0,t0, R1,t1)

    # Add noise
    random.seed(123)
    pts = random.randn(NUM_CORRESPONDENCES,3) + array([0., 0., -5.])
    xs0 = array([ pr(dot(K, dot(R0, x) + t0)) for x in pts ])
    xs1 = array([ pr(dot(K, dot(R1, x) + t1)) for x in pts ])
    xs0 += random.randn(*xs0.shape) * noise
    xs1 += random.randn(*xs1.shape) * noise

    # Perturb R and t so that we do not evaluate jacobians at the minima
    R  = dot(R, SO3.exp(random.randn(3) * .05))
    t += random.randn(3) * .05

    return R, t, xs0, xs1


################################################################################
# Test jacobians of the sampson error
class FundamentalMatrixTest(NumpyTestCase):
    def setUp(self):
        self.R, self.t, self.xs0, self.xs1 = setup_test_problem()
        self.x0 = unpr(self.xs0[0])
        self.x1 = unpr(self.xs1[0])
        self.x = array([ .6,  -2.,  1. ])
        pass

    def test_cauchy(self):
        r = array([1., 3., -2.])
        J = eye(3)
        self.assertJacobian(cauchy_sqrtcost_from_residual_multidimensional,
                            Jcauchy_sqrtcost_from_residual_multidimensional(r, J),
                            r)

        x0 = arange(4) - 2.5
        f  = lambda x: x**2 + x
        Jf = lambda x: diag(2.*x + 1)
        c  = lambda x: cauchy_sqrtcost_from_residual_multidimensional(f(x))
        Jc = lambda x: Jcauchy_sqrtcost_from_residual_multidimensional(f(x), Jf(x))
        self.assertJacobian(f, Jf, x0)
        self.assertJacobian(c, Jc, x0)

    def test_cauchy_hessian(self):
        J = np.arange(6).reshape((3,2)).astype(float)
        y = arange(3)[::-1].astype(float)
        f = lambda x: dot(J, x) - y
        E = lambda x: cauchy_cost(f(x))

        x0 = arange(2)+2.5
        r0 = f(x0)

        self.assertJacobian(f, J, x0)
        self.assertJacobian(f, J, x0)
        
        H_analytic = cauchy_hessian_firstorder(r0, J)
        print H_analytic

        H_numeric = finite_differences.numeric_hessian(E, x0)[0]
        print H_numeric

        self.assertArrayEqual(H_analytic, H_numeric)


    def test_JF1x(self):
        R,t,x = self.R, self.t, self.x
        fR = lambda m:  F1x(K, dot(R, SO3.exp(m)), t, x)
        ft = lambda tt: F1x(K, R, tt, x)
        self.assertJacobian(fR, JF1x_R(K, R, t, x)[newaxis,:], zeros(3))
        self.assertJacobian(ft, JF1x_t(K, R, t, x)[newaxis,:], t)

    def test_JFT1x(self):
        R,t,x = self.R, self.t, self.x
        fR = lambda m: FT1x(K, dot(R, SO3.exp(m)), t, x)
        ft = lambda tt: FT1x(K, R, tt, x)
        self.assertJacobian(fR, JFT1x_R(K, R, t, x)[newaxis,:], zeros(3))
        self.assertJacobian(ft, JFT1x_t(K, R, t, x)[newaxis,:], t)

    def test_JF2x(self):
        R,t,x = self.R, self.t, self.x
        fR = lambda m:  F2x(K, dot(R, SO3.exp(m)), t, x)
        ft = lambda tt: F2x(K, R, tt, x)
        self.assertJacobian(fR, JF2x_R(K, R, t, x)[newaxis,:], zeros(3))
        self.assertJacobian(ft, JF2x_t(K, R, t, x)[newaxis,:], t)

    def test_JFT2x(self):
        R,t,x = self.R, self.t, self.x
        fR = lambda m: FT2x(K, dot(R, SO3.exp(m)), t, x)
        ft = lambda tt: FT2x(K, R, tt, x)
        self.assertJacobian(fR, JFT2x_R(K, R, t, x)[newaxis,:], zeros(3))
        self.assertJacobian(ft, JFT2x_t(K, R, t, x)[newaxis,:], t)

    def test_JxFx(self):
        R,t,x0,x1 = self.R, self.t, self.x0, self.x1
        fR = lambda m: xFx(K, dot(R, SO3.exp(m)), t, x0, x1)
        ft = lambda tt: xFx(K, R, tt, x0, x1)
        self.assertJacobian(fR, JxFx_R(K, R, t, x0, x1)[newaxis,:], zeros(3))
        self.assertJacobian(ft, JxFx_t(K, R, t, x0, x1)[newaxis,:], t)

    def test_Jresidual(self):
        R,t,x0,x1 = self.R, self.t, self.x0, self.x1
        f = lambda v: residual(K, dot(R, SO3.exp(v[:3])), t+v[3:], x0, x1)
        self.assertJacobian( f, Jresidual(K, R, t, x0, x1)[newaxis,:], zeros(6) )

    def test_Jresidual_robust(self):
        R,t,x0,x1 = self.R, self.t, self.x0, self.x1
        f = lambda v: residual_robust(K, dot(R, SO3.exp(v[:3])), t+v[3:], x0, x1)
        self.assertJacobian( f, Jresidual_robust(K, R, t, x0, x1)[newaxis,:], zeros(6) )

    def test_normal_equations(self):
        R,t,xs0,xs1 = self.R, self.t, self.xs0, self.xs1
        xs0 = unpr(xs0)
        xs1 = unpr(xs1)
        c = lambda v: cost(K, dot(R, SO3.exp(v[:3])), t+v[3:], xs0, xs1)
        JTJ,JTr = compute_normal_equations(K, R, t, xs0, xs1, residual, Jresidual)
        H = finite_differences.numeric_hessian(c, zeros(6))
        print 'numeric hessian (least squares):'
        print H
        print '2*JTJ (least squares):'
        print 2.*JTJ
        self.assertJacobian( c, 2.*JTr, zeros(6) )

    def test_normal_equations_robust(self):
        R,t,xs0,xs1 = self.R, self.t, self.xs0, self.xs1
        xs0 = unpr(xs0)
        xs1 = unpr(xs1)
        c = lambda v: cost_robust(K, dot(R, SO3.exp(v[:3])), t+v[3:], xs0, xs1)
        JTJ,JTr = compute_normal_equations(K, R, t, xs0, xs1, residual_robust, Jresidual_robust)
        H = finite_differences.numeric_hessian(c, zeros(6))
        print 'numeric hessian (robust):'
        print H
        print '2*JTJ (robust):'
        print 2.*JTJ
        self.assertJacobian( c, 2.*JTr, zeros(6) )

if __name__ == '__main__':
    seterr(all='raise')
    unittest.main()
