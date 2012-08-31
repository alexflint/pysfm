import sys
from numpy import *
from numpy.linalg import *

from helpers import *

################################################################################
NUM_CORRESPONDENCES = 50
SENSOR_NOISE        = 0.1

MAX_STEPS           = 15
CONVERGENCE_THRESH  = 1e-5   # convergence detected when improvement less than this
INIT_DAMPING        = .1

K                   = eye(3)
Kinv                = inv(K)


################################################################################
# Compute the fundamental matrix K^-T * R * skew(x) * K^-1
def fund(K, R, t):
    return dots(inv(K).T, skew(t), R, inv(K))


################################################################################
def F1x(K, R, t, x):
    F = fund(K, R, t)
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
    F = fund(K, R, t)
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
    F = fund(K, R, t)
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
    F = fund(K, R, t)
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
    return dots(x1, fund(K, R, t), x0)

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
    m = zeros(3)

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
def cost(K, R, t, xs0, xs1):
    c = 0.
    for x0,x1 in zip(xs0,xs1):
        c += square(residual(K, R, t, x0, x1))
    return c

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

    xs0 = unpr(xs0)   # convert a list of 2-vectors to a list of 3-vectors with the last element set to 1
    xs1 = unpr(xs1)
    cost_cur = cost(K, R_cur, t_cur, xs0, xs1)

    while num_steps < MAX_STEPS and damping < 1e+8 and not converged:
        print 'Step %d:   cost=%-10f damping=%f' % (num_steps, cost_cur, damping)

        # Pick a parameter to freeze
        # TODO: do this the proper way instead
        freeze = 3 + argmax(abs(t_cur))
        mask = arange(6) != freeze

        # Compute J^T * J and J^T * r
        JTJ = zeros((6,6))  # 6x6
        JTr = zeros(6)      # 6x1
        for x0,x1 in zip(xs0,xs1):
            # Jacobian for the i-th point:
            ri =  residual(K, R_cur, t_cur, x0, x1)
            Ji = Jresidual(K, R_cur, t_cur, x0, x1)
            # Add to full jacobian
            JTJ += outer(Ji, Ji)    # 5x1 * 1x5 -> 5x5
            JTr +=   dot(Ji, ri)    # 5x1 * 1x1 -> 5x1

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
            R_next = dot(R_cur, SO3_exp(update[:3]))
            t_next = t_cur + update[3:]

            # Compute new cost
            cost_next = cost(K, R_next, t_next, xs0, xs1)
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

    if converged:
        print 'CONVERGED AFTER %d STEPS' % num_steps

    return R_cur, t_cur









################################################################################
def setup_test_problem():
    R0 = eye(3)
    t0 = zeros(3)

    R1 = dots(rotation_xz(-.2), rotation_xy(.1), rotation_yz(1.5))
    t1 = array([-1., .5, -2.])
    
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

    random.seed(73)
    R_init = dot(R_true, SO3_exp(random.randn(3)*.2))
    t_init = t_true + random.randn(3)*.1

    savetxt('essential_matrix_data/init_pose.txt', hstack((R_init, t_init[:,newaxis])), fmt='%10f')
    savetxt('essential_matrix_data/100_correspondences_tenpercent_noise.txt', hstack((xs0, xs1)), fmt='%10f')
    
    R_opt, t_opt = optimize_fmatrix(xs0, xs1, R_init, t_init)

    print '\nTrue [R t]:'
    print hstack((R_true, t_true[:,newaxis]))
    print '\nInitial [R t]:'
    print hstack((R_init, t_init[:,newaxis]))
    print '\nFinal [R t]:'
    print hstack((R_opt, t_opt[:,newaxis]))
    

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

    R_opt, t_opt = optimize_fmatrix(xs0, xs1, R_init, t_init)

    print '\nInitial [R t]:'
    print hstack((R_init, t_init[:,newaxis]))
    print '\nFinal [R t]:'
    print hstack((R_opt, t_opt[:,newaxis]))


################################################################################
if __name__ == '__main__':
    #run_with_synthetic_data()
    run_with_data_from_file()
