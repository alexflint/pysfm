import sys
import itertools
from numpy import *
from numpy.linalg import *
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from algebra import *
import lie
import homography_sampson_helpers
import homography

################################################################################

# Optimization parameters
MAX_STEPS           = 50
CONVERGENCE_THRESH  = 1e-5   # convergence detected when improvement less than this
INIT_DAMPING        = .1
FREEZE_LARGEST      = True

# Objective function parameters
CAUCHY_SIGMA        = .1
CAUCHY_SIGMA_SQR    = CAUCHY_SIGMA * CAUCHY_SIGMA
DEFAULT_ERROR_FUNC  = 'transfer'
SL3_PARAMETRISATION = True

# Parameters for generating synthetic data
NUM_CORRESPONDENCES =  20
SENSOR_NOISE        = .00
PERTURBATION        = .002
SKEW_MARGIN         = .5

################################################################################
# Compute the algebraic error. Used only for sanity checking during debugging
def algebraic_error(H, x0, x1):
    return dot(H[:2], unpr(x0)) - dot(H[2], unpr(x0)) * x1[:2]

# Compute the jacobian of the algebraic error w.r.t. x0 and x1. Used
# only for sanity checking during debugging
def algebraic_error_jacobian(H, x0, x1):
    M = H[:2,:2]
    g = H[ 2,:2]
    l = H[ 2, 2]
    return hstack(( M-outer(x1,g), -(dot(g,x0) + l)*eye(2) ))


################################################################################
def sampson_error(H, x0, x1):
    return homography_sampson_helpers.error(x0, x1, H)

def sampson_error_jacobian(H, x0, x1):
    return homography_sampson_helpers.jacobian(x0, x1, H)

def sampson_error_sl3_jacobian(H, x0, x1):
    JH = vstack(( dot(H, lie.SL3_BASIS[i]).reshape((1,9)) for i in range(8) )).T
    Jf = homography_sampson_helpers.jacobian(x0, x1, H)
    return dot(Jf, JH)


################################################################################
# One-way transfer error
def transfer_error(H, x0, x1):
    return prdot(H, x0) - x1

# One-way transfer error jacobian (w.r.t. matrix elements)
def transfer_error_jacobian(H, x0, x1):
    y = dot(H, unpr(x0))
    Jpr = array([[ 1./y[2],    0,         -y[0] / (y[2]*y[2])],
                 [ 0,          1./y[2],   -y[1] / (y[2]*y[2])]])
    JHx = zeros((3,9))
    JHx[0, 0:3] = unpr(x0)
    JHx[1, 3:6] = unpr(x0)
    JHx[2, 6:9] = unpr(x0)
    return dot(Jpr, JHx)

# One-way transfer error jacobian (w.r.t. SL3 coefficients)
def transfer_error_sl3_jacobian(H, x0, x1):
    JH = vstack((dot(H, lie.SL3_BASIS[i]).reshape((1,9)) for i in range(8))).T
    Jf = transfer_error_jacobian(H, x0, x1)
    return dot(Jf, JH)


################################################################################
# Symmetric transfer error
def symtransfer_error(H, x0, x1):
    return hstack((transfer_error(H, x0, x1),
                   transfer_error(inv(H), x1, x0)))

# Symmetric transfer error jacobian
def symtransfer_error_jacobian(H, x0, x1):
    return vstack((transfer_error_jacobian(H, x0, x1),
                   transfer_error_jacobian(inv(H), x1, x0)))


################################################################################
def cauchy_cost(r):
    return log( 1. + dot(r,r) / CAUCHY_SIGMA_SQR )

def cauchy_jacobian(r):
    C   = cauchy_cost(r)
    JTJ = dot(J.T, J)
    g   = dot(J.T, r)
    d   = CAUCHY_SIGMA_SQR + dot(r,r)
    return 2*JTJ/d - 4*outer(g,g)/(d*d)

def cauchy_hessian_firstorder(r, J):
    JTJ = dot(J.T, J)
    g   = dot(J.T, r)
    d   = CAUCHY_SIGMA_SQR + dot(r,r)
    return 2.*JTJ/d - 4.*outer(g,g)/(d*d)

# Given a residual vector, compute the square root of the
# Cauchy-robustified error response.
def cauchy_sqrtcost_from_residual(r):
    return sqrt(log(1. + r*r / CAUCHY_SIGMA_SQR))

# Given a residual vector and its gradient, compute the derivative of
# the square root of the Cauchy-robustified error response.
def Jcauchy_sqrtcost_from_residual(r, dr):
    assert isscalar(r)
    assert ndim(dr) == 1
    sqrtcost = cauchy_sqrtcost_from_residual(r)
    if sqrtcost < 1e-8:
        return 0.     # THIS WILL COME UP OFTEN: in particular
                      # whenever the current F agrees with a
                      # correspondence pair
                      # this is in fact the correct thing to do here
    return dr * r / (sqrtcost * (CAUCHY_SIGMA_SQR + r*r))




################################################################################
def residual(H, x0, x1, error=DEFAULT_ERROR_FUNC):
    if error == 'transfer':
        return transfer_error(H, x0, x1)
    elif error == 'symtransfer':
        return symtransfer_error(H, x0, x1)
    elif error == 'sampson':
        return sampson_error(H, x0, x1)
    else:
        raise Exception, 'Unknown error function: '+error

def Jresidual(H, x0, x1, error=DEFAULT_ERROR_FUNC):
    if SL3_PARAMETRISATION:
        if error == 'transfer':
            return transfer_error_sl3_jacobian(H, x0, x1)
        elif error == 'symtransfer':
            return symtransfer_error_sl3_jacobian(H, x0, x1)
        elif error == 'sampson':
            return sampson_error_sl3_jacobian(H, x0, x1)
        else:
            raise Exception, 'Unknown error: '+error
    else:
        if error == 'transfer':
            return transfer_error_jacobian(H, x0, x1)
        elif error == 'symtransfer':
            return symtransfer_error_jacobian(H, x0, x1)
        elif error == 'sampson':
            return sampson_error_jacobian(H, x0, x1)
        else:
            raise Exception, 'Unknown error function: '+error

def residual_robust(H, x0, x1):
    #return cauchy_sqrtcost_from_residual(residual(K, R, t, x0, x1))
    raise Exception('Not implemented')

def Jresidual_robust(H, x0, x1):
    #return Jcauchy_sqrtcost_from_residual(residual(K, R, t, x0, x1),
    #                                      Jresidual(K, R, t, x0, x1))
    raise Exception('Not implemented')


################################################################################
def cost(H, xs0, xs1, error=DEFAULT_ERROR_FUNC):
    return sum(sum(square(residual(H, x0, x1, error))
                   for x0,x1 in zip(xs0,xs1)))

def cost_robust(K, R, t, xs0, xs1):
    return sum(sum(square(residual_robust(H, x0, x1))
                   for x0,x1 in zip(xs0,xs1)))

################################################################################
# Compute J^T * J and J^T * r
import finite_differences
def compute_normal_equations(H, xs0, xs1, error=DEFAULT_ERROR_FUNC):
    num_params = 8 if SL3_PARAMETRISATION else 9

    JTJ = zeros((num_params, num_params))
    JTr = zeros(num_params)

    for x0,x1 in zip(xs0,xs1):
        # Jacobian for the i-th point:
        ri =  residual(H, x0, x1, error)
        Ji = Jresidual(H, x0, x1, error)

        #fi = lambda w: residual(dot(H, lie.SL3_exp(w)), x0, x1, error)
        #finite_differences.check_jacobian(fi, Ji, zeros(8))

        # Add to full jacobian
        Ji = atleast_2d(Ji)     # so that dot product reduces to outer product for vectors
        ri = atleast_1d(ri)
        #print error
        #print 'ri:',np.shape(ri)
        #print 'Ji:',Ji.shape
        #print dot(Ji.T,Ji).shape
        JTJ += dot(Ji.T, Ji)    # 9x1 * 1x9 -> 9x9
        JTr += dot(Ji.T, ri)    # 9x1 * 1x1 -> 9x1

    return JTJ, JTr

################################################################################
# Given:
#   xs0    -- a list of 2D points in view 0
#   xs1    -- a list of 2D points in view 1
#   R_init -- an initial rotation between view 0 and view 1
#   t_init -- an initial translation between view 0 and view 1
# Find the essential matrix relating the two views by minimizing the
# Sampson error.
def optimize_homography(xs0, xs1, H_init, error=DEFAULT_ERROR_FUNC):
    num_steps = 0
    converged = False
    damping = INIT_DAMPING
    H_cur = H_init.copy()
    H_cur /= norm(H_cur)

    cost_cur = cost(H_cur, xs0, xs1, error)

    # Create a function that returns a new function that linearizes costfunc about (R,t)
    # This is used for debugging only
    #flocal = lambda R,t: lambda v: costfunc(K, dot(R, SO3.exp(v[:3])), t+v[3:], xs0, xs1)

    while num_steps < MAX_STEPS and damping < 1e+8 and not converged:
        print 'Step %d:   cost=%-10f damping=%f' % (num_steps, cost_cur, damping)

        # Compute normal equations
        JTJ,JTr = compute_normal_equations(H_cur, xs0, xs1, error)

        # Look for a valid step length
        while not converged:
            # Check for failure
            if damping > 1e+8:
                print 'FAILED TO CONVERGE (due to large damping)'
                break

            # Apply Levenberg-Marquardt damping
            b = JTr.copy()
            A = JTJ.copy()
            A[ diag_indices(len(A)) ] *= (1. + damping)

            # Solve normal equations: 6x6
            try:
                update = -solve(A, b)
                #print update
            except LinAlgError:
                # Badly conditioned: increase damping and try again
                print '  [failed to solve normal equations]'
                damping *= 10.
                continue

            # Take gradient step
            if SL3_PARAMETRISATION:
                #print lie.SL3_exp(update)
                H_next = dot(H_cur, lie.SL3_exp(update))
            else:
                H_next = H_cur + update.reshape((3,3))

            #H_next /= norm(H_next)

            # Compute new cost
            cost_next = cost(H_next, xs0, xs1, error)
            if cost_next < cost_cur:
                # Cost decreased: accept the update
                if cost_cur - cost_next < CONVERGENCE_THRESH:
                    print 'Converged due to small improvement'
                    converged = True

                # Do not decrease damping if in danger of hitting machine epsilon
                if damping > 1e-15:
                    damping *= .1

                H_cur = H_next.copy()
                cost_cur = cost_next
                num_steps += 1
                break

            else:
                # Cost increased: reject the udpate, increase damping
                #print '  [step caused an increase in the cost function (%f --> %f)]' % \
                #    (cost_cur, cost_next)
                damping *= 10.

    if converged:
        print 'CONVERGED AFTER %d STEPS' % num_steps

    return H_cur



################################################################################
def setup_test_problem(n=NUM_CORRESPONDENCES):
    H = array([[ 1.5,  .1,  -2.  ],
               [ 0.,   .8,   1.1 ],
               [ 0.,  0. ,   1.  ]])

    random.seed(123)
    xs0_true  = random.randn(n, 2)
    xs1_true  = prdot(H, xs0)
    xs0 = xs0_true + random.randn(*xs0_true.shape) * SENSOR_NOISE
    xs1 = xs1_true + random.randn(*xs1_true.shape) * SENSOR_NOISE

    return H, xs0, xs1, xs0_true, xs1_true

################################################################################
def setup_skewed_test_problem(n=NUM_CORRESPONDENCES):
    k = SKEW_MARGIN
    quad0 = [(0,0), (0,10), (10,0), (10,10)]
    quadL = [(0,0), (0,10), (3,5-k),  (3,5+k)  ]
    quadR = [(7,5-k), (7,5+k),  (10,0), (10,10)]

    HL = homography.estimate(quad0, quadL)
    HR = homography.estimate(quad0, quadR)
    H = dot(HR, inv(HL))

    random.seed(123)
    xs0 = array(list(itertools.product([0,10],linspace(0,10,15))), float)
    #xs0 = random.uniform(0, 10, (n,2))

    xsL_true = prdot(HL, xs0)
    xsR_true = prdot(HR, xs0)
    
    xsL = xsL_true + random.randn(*xsL_true.shape) * SENSOR_NOISE
    xsR = xsR_true + random.randn(*xsR_true.shape) * SENSOR_NOISE

    return H, xsL, xsR, xsL_true, xsR_true

################################################################################
def draw_points(points, *args, **kwargs):
    plt.plot(points[:,0], points[:,1], *args, **kwargs)

def draw_polygon(path, *args, **kwargs):
    xs,ys = zip(*path)
    xs = list(xs)
    ys = list(ys)
    xs.append(xs[0])
    ys.append(ys[0])
    plt.plot(xs, ys, *args, **kwargs)

def draw_optimization_results(H_true, H_init, H_opt, xs0, xs1, xs0_true, xs1_true):
    loffs = array([0., 0.])
    roffs = array([0., 0.])

    k = SKEW_MARGIN
    quadL = [(0,0), (0,10), (3,5+k), (3,5-k)  ]

    draw_polygon(loffs+quadL, 'r')
    #draw_polygon(roffs+prdot(H_true, quadL), 'r')
    draw_polygon(roffs+prdot(H_init, quadL), 'g')
    draw_polygon(roffs+prdot(H_opt, quadL), 'b')

    draw_points(loffs + xs0_true, '.r')
    draw_points(loffs + xs0, 'xr')

    draw_points(roffs + xs1_true, '.g')
    draw_points(roffs + xs1, 'xg')

    draw_points(roffs + prdot(H_true, xs0), 'xr')
    draw_points(loffs + prdot(inv(H_true), xs1), 'xg')

################################################################################
def run_with_synthetic_data():
    #H_true, xs0, xs1, xs0_true, xs1_true = setup_test_problem()
    H_true, xs0, xs1, xs0_true, xs1_true = setup_skewed_test_problem()
    H_true /= H_true[2,2]

    random.seed(73)
    H_init = dot(eye(3) + random.randn(3,3)*PERTURBATION, H_true)
    H_init /= H_init[2,2]
    
    #savetxt('standalone/essential_matrix_data/true_pose.txt',
    #        hstack((R_true, t_true[:,newaxis])), fmt='%10f')

    pdf = PdfPages('out/homography.pdf')
    for error_func in ('transfer', 'sampson'):
        H_opt = optimize_homography(xs0, xs1, H_init, error_func)
        H_opt /= H_opt[2,2]

        print '\nTrue H:'
        print H_true
        print '\nInitial H:'
        print H_init
        print '\nFinal H:'
        print H_opt

        print '\nError in H:'
        print dot(inv(H_opt), H_true)

        plt.clf()
        draw_optimization_results(H_true, H_init, H_opt, xs0, xs1, xs0_true, xs1_true)
        plt.title(error_func)
        pdf.savefig()

    pdf.close()

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

    t_init /= norm(t_init)

    print '\nInitial [R t]:'
    print hstack((R_init, t_init[:,newaxis]))
    print '\nFinal [R t]:'
    print hstack((R_opt, t_opt[:,newaxis]))


################################################################################
if __name__ == '__main__':
    # avoid ugly scientific notation inside arrays)
    np.set_printoptions(suppress=True)

    run_with_synthetic_data()
    #run_with_data_from_file()
    #foo()
