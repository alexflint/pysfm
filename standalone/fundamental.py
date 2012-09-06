import sys
from numpy import *
from numpy.linalg import *
import numpy as np

from helpers import *

# Notes:
#  - x0 and x1 are always points in two different images. The
#    convention throughout is that all fundamental matrices
#    are "from" image 0 (which contains x0) "to" image 1 (which
#    contains x1). i.e.:
#       x1.T * F * x0 = 0
#  - Hence the epipolar line for x0 in image 1 is F.T*x0 and the
#    epipolar line for x1 in image 0 is F*x1

################################################################################
NUM_CORRESPONDENCES = 500
SENSOR_NOISE        = 0.01

MAX_STEPS           = 15
CONVERGENCE_THRESH  = 1e-5   # convergence detected when improvement less than this
INIT_DAMPING        = .1

K                   = eye(3)
Kinv                = inv(K)

FREEZE_LARGEST      = True

################################################################################
# Compute the Fundamental matrix K^-T * R * skew(x) * K^-1
def make_fundamental(K, R, t):
    return dots(Kinv.T, skew(t), R, Kinv)

################################################################################
# Performs an "inverse" masking operation on a vector
def unmask(v, mask, fill=0.):
    assert mask.dtype.kind == 'b'
    assert np.sum(mask) == len(v)
    u = zeros(len(mask))
    u[mask] = v
    return u

################################################################################
def point_line_distance(p, l):
    assert shape(p) == (3,)
    assert shape(l) == (3,)
    return abs(dot(p,l) / (p[2] * norm(l[:2])))

################################################################################
# Compute the algebraic error. Used only for sanity checking during debugging.
def algebraic_error_forwards(K, R, t, xs0, xs1):
    F = make_fundamental(K, R, t)
    c = 0.
    for x0,x1 in zip(xs0,xs1):
        c += point_line_distance(dot(F, unpr(x0)), unpr(x1))
    return c

################################################################################
def F1x(K, R, t, x):
    F = make_fundamental(K, R, t)
    return dot(F[0], x)

def JF1x_R(K, R, t, x):
    v = dots(R.T, skew(t), Kinv[:,0])
    return dots(skew(v), Kinv, x)

def JF1x_t(K, R, t, x):
    v = Kinv[:,0]
    return -dots(skew(v), R, Kinv.T, x)

def JF1x(K, R, t, x):
    return concatenate((JF1x_R(K, R, t, x),
                        JF1x_t(K, R, t, x)))


################################################################################
def FT1x(K, R, t, x):
    F = make_fundamental(K, R, t)
    return dot(F[:,0], x)

def JFT1x_R(K, R, t, x):
    v = Kinv[:,0]
    return dots(skew(v), R.T, skew(-t), Kinv, x)

def JFT1x_t(K, R, t, x):
    v = dot(R, Kinv[:,0])
    return dots(skew(v), Kinv, x)

def JFT1x(K, R, t, x):
    return concatenate((JFT1x_R(K, R, t, x),
                        JFT1x_t(K, R, t, x)))


################################################################################
def F2x(K, R, t, x):
    F = make_fundamental(K, R, t)
    return dot(F[1], x)

def JF2x_R(K, R, t, x):
    v = dots(R.T, skew(t), Kinv[:,1])
    return dots(skew(v), Kinv, x)

def JF2x_t(K, R, t, x):
    v = Kinv[:,1]
    return -dots(skew(v), R, Kinv, x)

def JF2x(K, R, t, x):
    return concatenate((JF2x_R(K, R, t, x),
                        JF2x_t(K, R, t, x)))


################################################################################
def FT2x(K, R, t, x):
    F = make_fundamental(K, R, t)
    return dot(F[:,1], x)

def JFT2x_R(K, R, t, x):
    v = Kinv[:,1]
    return dots(skew(v), R.T, skew(-t), Kinv, x)

def JFT2x_t(K, R, t, x):
    v = dot(R, Kinv[:,1])
    return dots(skew(v), Kinv, x)

def JFT2x(K, R, t, x):
    return concatenate((JFT2x_R(K, R, t, x),
                        JFT2x_t(K, R, t, x)))


################################################################################
def xFx(K, R, t, x0, x1):
    assert shape(K) == (3,3)
    assert shape(R) == (3,3)
    assert shape(t) == (3,)
    assert shape(x0) == (3,)
    assert shape(x1) == (3,)
    return dots(x1, make_fundamental(K, R, t), x0)

def JxFx_R(K, R, t, x0, x1):
    v = dot(Kinv, x0)
    return dots(x1, Kinv.T, skew(t), R, skew(-v))

def JxFx_t(K, R, t, x0, x1):
    v = dots(R, Kinv, x0)
    return dots(x1, Kinv.T, skew(-v))

def JxFx(K, R, t, x0, x1):
    return concatenate((JxFx_R(K, R, t, x0, x1),
                        JxFx_t(K, R, t, x0, x1)))


################################################################################
def residual(K, R, t, x0, x1):
    f  = xFx (K, R, t, x0, x1)
    g1 = F1x (K, R, t, x0)
    g2 = F2x (K, R, t, x0)
    g3 = FT1x(K, R, t, x1)
    g4 = FT2x(K, R, t, x1)
    return f / sqrt(g1*g1 + g2*g2 + g3*g3 + g4*g4)

def Jresidual(K, R, t, x0, x1):
    f = xFx(K, R, t, x0, x1)
    Jf = JxFx(K, R, t, x0, x1)

    g1 = F1x(K, R, t, x0)
    Jg1 = JF1x(K, R, t, x0)

    g2 = F2x(K, R, t, x0)
    Jg2 = JF2x(K, R, t, x0)

    g3 = FT1x(K, R, t, x1)
    Jg3 = JFT1x(K, R, t, x1)

    g4 = FT2x(K, R, t, x1)
    Jg4 = JFT2x(K, R, t, x1)

    d = g1*g1 + g2*g2 + g3*g3 + g4*g4
    Jd = g1*Jg1 + g2*Jg2 + g3*Jg3 + g4*Jg4
    J = Jf/sqrt(d) - f*Jd / (d * sqrt(d))
    return J

################################################################################
def residual_robust(K, R, t, x0, x1):
    return cauchy_sqrtcost_from_residual(residual(K, R, t, x0, x1))

def Jresidual_robust(K, R, t, x0, x1):
    return Jcauchy_sqrtcost_from_residual(residual(K, R, t, x0, x1),
                                          Jresidual(K, R, t, x0, x1))

################################################################################
def cost_robust(K, R, t, xs0, xs1):
    c = 0.
    for x0,x1 in zip(xs0,xs1):
        c += square(residual_robust(K, R, t, x0, x1))
    return c

################################################################################
# Compute JT*J and JT*r
def compute_normal_equations(K, R, t, xs0, xs1):
    # Compute J^T * J and J^T * r
    JTJ = zeros((6,6))  # 6x6
    JTr = zeros(6)      # 6x1

    for x0,x1 in zip(xs0,xs1):
        # Jacobian for the i-th point:
        ri =  residual_robust(K, R, t, x0, x1)
        Ji = Jresidual_robust(K, R, t, x0, x1)
        # Add to full jacobian
        JTJ += outer(Ji, Ji)    # 6x1 * 1x6 -> 6x6
        JTr +=   dot(Ji, ri)    # 6x1 * 1x1 -> 6x1

    # Returns: a 6x6 matrix and a 6x1 vector
    return JTJ, JTr

################################################################################
# Given:
#   xs0    -- a list of 2D points in view 0
#   xs1    -- a list of 2D points in view 1
#   R_init -- an initial rotation between view 0 and view 1
#   t_init -- an initial translation between view 0 and view 1
# Find the essential matrix relating the two views by minimizing the
# Sampson error.
def optimize_fmatrix(xs0, xs1, R_init, t_init):
    num_steps = 0
    converged = False
    damping = INIT_DAMPING
    R_cur = R_init.copy()
    t_cur = t_init.copy()
    t_cur /= norm(t_cur)

    xs0 = unpr(xs0)   # convert a list of 2-vectors to a list of 3-vectors with the last element set to 1
    xs1 = unpr(xs1)
    cost_cur = cost_robust(K, R_cur, t_cur, xs0, xs1)

    while num_steps < MAX_STEPS and damping < 1e+8 and not converged:
        print 'Step %d:   cost=%-10f damping=%f' % (num_steps, cost_cur, damping)

        # Compute normal equations
        JTJ,JTr = compute_normal_equations(K, R_cur, t_cur, xs0, xs1)

        # Pick a step length
        while damping < 1e+8 and not converged:
            # Check for failure
            if damping > 1e+8:
                print 'FAILED TO CONVERGE'
                break

            # Apply Levenberg-Marquardt damping
            b = JTr.copy()
            A = JTJ.copy()
            for i in range(A.shape[0]):
                A[i,i] *= (1. + damping)

            # Freeze one translation parameter:
            #   1. remove its row and column from normal equations
            #   2. solve normal 5x5 normal equations
            #   3. insert a zero into the update where the parameter would have been
            if FREEZE_LARGEST:
                param_to_freeze = 3 + np.argmax(t_cur)
                mask = (arange(6) != param_to_freeze)
                A = A[mask].T[mask].T  # Select 5 rows and corresponding columns
                b = b[mask]            # Select the corresponding elements of RHS 
    
            # Solve normal equations
            try:
                update = -solve(A, b)
            except LinAlgError:
                # Badly conditioned: increase damping and try again
                damping *= 10.
                continue

            # Expand the update
            if FREEZE_LARGEST:
                update = unmask(update, mask)

            # Take gradient step
            R_next = dot(R_cur, SO3_exp(update[:3]))
            t_next = t_cur + update[3:]

            # Normalize to unit length
            t_next /= norm(t_next)

            # Compute new cost
            cost_next = cost_robust(K, R_next, t_next, xs0, xs1)
            if cost_next < cost_cur:
                # Cost decreased: accept the update
                if cost_cur - cost_next < CONVERGENCE_THRESH:
                    converged = True

                # Do not decrease damping if we're in danger of hitting machine epsilon
                if damping > 1e-15:
                    damping *= .1

                R_cur = R_next
                t_cur = t_next
                cost_cur = cost_next
                num_steps += 1
                break

            else:
                # Cost increased: reject the udpate, try another step length
                damping *= 10.

    if converged:
        print 'CONVERGED AFTER %d STEPS' % num_steps

    return R_cur, t_cur



################################################################################
def setup_test_problem():
    R0 = eye(3)
    t0 = zeros(3)

    R1 = dots(rotation_xz(-.2), rotation_xy(.1), rotation_yz(1.5))
    t1 = array([-1., .5, -2.])
    t1 /= norm(t1)
    
    R,t = relative_pose(R0,t0, R1,t1)

    random.seed(123)
    pts = random.randn(NUM_CORRESPONDENCES,3) + array([0., 0., -5.])
    xs0 = array([ pr(dot(K, dot(R0, x) + t0)) for x in pts ])
    xs1 = array([ pr(dot(K, dot(R1, x) + t1)) for x in pts ])
    xs0 += random.randn(*xs0.shape) * SENSOR_NOISE
    xs1 += random.randn(*xs1.shape) * SENSOR_NOISE

    return R, t, xs0, xs1


################################################################################
def run_with_synthetic_data():
    R_true, t_true, xs0, xs1 = setup_test_problem()
    #savetxt('standalone/essential_matrix_data/true_pose.txt',
    #        hstack((R_true, t_true[:,newaxis])), fmt='%10f')

    PERTURB = .1

    random.seed(73)
    R_init = dot(R_true, SO3_exp(random.randn(3)*PERTURB))
    t_init = t_true + random.randn(3)*PERTURB

    # savetxt('standalone/essential_matrix_data/init_pose.txt',
    #         hstack((R_init, t_init[:,newaxis])),
    #         fmt='%10f')
    # savetxt('standalone/essential_matrix_data/100_correspondences_tenpercent_noise.txt',
    #         hstack((xs0, xs1)),
    #         fmt='%10f')
    
    R_opt, t_opt = optimize_fmatrix(xs0, xs1, R_init, t_init)

    print '\nTrue [R t]:'
    print hstack((R_true, t_true[:,newaxis]))
    print '\nInitial [R t]:'
    print hstack((R_init, t_init[:,newaxis]))
    print '\nFinal [R t]:'
    print hstack((R_opt, t_opt[:,newaxis]))

    print '\nError in R:'
    print dot(R_opt.T, R_true) - eye(3)
    print '\nError in t:'
    print abs(t_opt - t_true)

    print '\nAlgebraic error at true:'
    print '  ',algebraic_error_forwards(K, R_true, t_true, xs0, xs1)
    print '\nAlgebraic error at initial:'
    print '  ',algebraic_error_forwards(K, R_init, t_init, xs0, xs1)
    print '\nAlgebraic error at final:'
    print '  ',algebraic_error_forwards(K, R_opt, t_opt, xs0, xs1)

################################################################################
def run_with_data_from_file():
    if len(sys.argv) != 3:
        print 'Usage: python %s CORRESPONDENCES.txt INITIAL_POSE.txt' % sys.argv[0]
        sys.exit(-1)

    corrs = loadtxt(sys.argv[1])
    xs0 = corrs[:,:2]
    xs1 = corrs[:,2:]

    P_init = loadtxt(sys.argv[2]).reshape((3,4))
    R_init = P_init[:,:3]
    t_init = P_init[:,3]
    t_init /= norm(t_init)

    R_opt, t_opt = optimize_fmatrix(xs0, xs1, R_init, t_init)

    print '\nInitial [R t]:'
    print hstack((R_init, t_init[:,newaxis]))
    print '\nFinal [R t]:'
    print hstack((R_opt, t_opt[:,newaxis]))


################################################################################
if __name__ == '__main__':
    #run_with_synthetic_data()
    run_with_data_from_file()
