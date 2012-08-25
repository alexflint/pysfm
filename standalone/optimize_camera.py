# Import python's numerics library
from numpy import *
from numpy.linalg import *
from numpy.random import *

# Notes on python numerics:
# A[i,:]   -- get the i-th row of A
# A[:,i]   -- get the i-th column of A
# A[:5]    -- get the first five rows of of A
# A[:,:2]  -- get the first two columns of of A

# Optimization parameters
MAX_STEPS = 100
CONVERGENCE_THRESH = 1e-5   # convergence detected when improvement less than this
INIT_DAMPING = 10.
CAUCHY_SIGMA = .1           # determines the steepness of the Cauchy robustifier near zero
CAUCHY_SIGMA_SQR = CAUCHY_SIGMA * CAUCHY_SIGMA
         
# Camera calibration
K = eye(3)

#
# Helpers
#

def dots3(A, B, C):
    return dot(A, dot(B, C))

def dots4(A, B, C, D):
    return dot(A, dot(B, dot(C,D)))

# Get the skew-symmetric matrix for m
def skew(m):
    return array([[  0,    -m[2],  m[1] ],
                  [  m[2],  0,    -m[0] ],
                  [ -m[1],  m[0],  0.   ]])

# Compute skew(m) * skew(m)
def skewsqr(m):
    a,b,c    = m
    aa,bb,cc = square(m)  # element-wise square
    return array([[ -bb-cc,   a*b,    a*c   ],
                  [  a*b,    -aa-cc,  b*c   ],
                  [  a*c,     b*c,   -aa-bb ]])

# Compute matrix exponential from so(3) to SO(3) (i.e. evaluate Rodrigues)
def SO3_exp(m):
    t = norm(m)
    if t < 1e-8:
        return eye(3)   # exp(0) = I

    a = sin(t)/t
    b = (1. - cos(t)) / (t*t)
    return eye(3) + a*skew(m) + b*skewsqr(m)

#
# Robustifiers
#

# Given a 2x1 reprojection error (in raw pixel coordinates), evaluate
# a robust error function and return a scalar cost.
def cauchy_cost_from_reprojection_error(x):
    x = asarray(x)
    return log(1. + (x[0]*x[0] + x[1]*x[1]) / CAUCHY_SIGMA_SQR)

# Given a 2x1 reprojection error (in raw pixel coordinates), evaluate
# a robust error function and return:
#   -- the 2x1 residual vector
#   -- the 2x2 Jacobian of that residual
def cauchy_residual_from_reprojection_error(x):
    x = asarray(x)
    rr = x[0]*x[0] + x[1]*x[1]
    if rr < 1e-8:   # within this window the residual is well-approximated as linear
        return x / CAUCHY_SIGMA

    r = sqrt(rr)
    e = sqrt(log(1. + rr / CAUCHY_SIGMA_SQR))
    xx = outer(x, x)
    residual = x * e / r
    Jresidual = xx / (r*e*(rr + CAUCHY_SIGMA_SQR)) + (r*eye(2) - xx/r) * e/rr
    return residual, Jresidual

#
# Jacobians
#

# Compute the reprojection error of x given a camera (R,t) with
# intrinsics K. Also compute the Jacobian with respect to R and t.
#   K -- intrinsics
#   R -- 3x3 camera rotation
#   t -- 3x1 camera translation
#   x -- 3x1 point
#   measurement -- 2x1 measured position of x in camera
#
# Let f(K, R, t, x) = pr(K * (R * x + t))
#   where pr(x,y,z) = [ x/z, y/z ]
#
# Then the Jacobian of f with respect to t is:
#   Jf_t = Jpr(K * (R * x + t)) * K
# and the Jacobian of f with respect to R is:
#   Jf_R = Jpr(K * (R * x + t)) * K
# where Jpr is the Jacobian of (x,y,z) -> (x/z, y/z)
def compute_camera_jacobian(K, R, t, x, measurement):
    # projection: p = K * (R*x + t)
    p = dot(K, dot(R, x) + t)
    # jacobian of projection with respect to p
    Jpr = array([[ 1./p[2],  0.,       -p[0] / (p[2]*p[2]) ],
                 [ 0.,       1./p[2],  -p[1] / (p[2]*p[2]) ]])
    # Jacobian of the projection with respect to rotation
    J_R = dots4(Jpr, K, R, skew(-x))
    # Jacobian of the projection with respect to transation
    J_t = dot(Jpr, K)
    # Concatenate
    J = hstack((J_R, J_t))
    # Divide by z
    prediction = p[:2] / p[2]
    return J, prediction - measurement

def compute_camera_jacobian_robust(K, R, t, x, measurement):
    # projection: p = K * (R*x + t)
    p = dot(K, dot(R, x) + t)
    # jacobian of projection with respect to p
    Jpr = array([[ 1./p[2],  0.,       -p[0] / (p[2]*p[2]) ],
                 [ 0.,       1./p[2],  -p[1] / (p[2]*p[2]) ]])

    # Divide by z
    prediction = p[:2] / p[2]
    reproj_error = prediction - measurement
    # Robustify
    residual, Jresidual = cauchy_residual_from_reprojection_error(reproj_error)
    Jpr = dot(Jresidual, Jpr)

    # Jacobian of the projection with respect to rotation
    J_R = dots4(Jpr, K, R, skew(-x))
    # Jacobian of the projection with respect to transation
    J_t = dot(Jpr, K)
    # Concatenate
    J = hstack((J_R, J_t))

    return J, residual

# Compute the sum of squared reprojection errors
def cost(points, measurements, K, R, t):
    error = 0.
    for i in range(len(points)):
        projection = dot(K, dot(R, points[i]) + t)
        projection = projection[:2] / projection[2]
        reproj_error = projection - measurements[i]
        error += sum(square(reproj_error))  # sum of squared differences
    return error

# Compute the sum of Cauchy-robustified reprojection errors
def cost_robust(points, measurements, K, R, t):
    error = 0.
    for i in range(len(points)):
        projection = dot(K, dot(R, points[i]) + t)
        projection = projection[:2] / projection[2]
        reproj_error = projection - measurements[i]
        error += cauchy_cost_from_reprojection_error(reproj_error)  # sum of squared differences
    return error


#
# Optimization
#

# Given:
#   points       -- a list of 3-vectors
#   measurements -- a list of 2-vectors
#   K            -- 3x3 camera intrinsics
#   R_init       -- 3x3 rotation, initializes camera rotation
#   t_init       -- 3-vector, initializes camera translation
# Find a new rotation and translation minimizing:
#   cost(points, measurements, K, R, t)
def optimize_pose(points, measurements, K, R_init, t_init):
    num_steps = 0
    R_cur = R_init
    t_cur = t_init
    cost_cur = cost_robust(points, measurements, K, R_cur, t_cur)
    converged = False
    damping = INIT_DAMPING
    while num_steps < MAX_STEPS and damping < 1e+8 and not converged:
        # Print status
        print 'Step %d: cost=%f' % (num_steps, cost_cur)

        # Compute J^T * J and J^T * r
        JTJ = zeros((6,6))
        JTr = zeros(6)
        for i in range(len(points)):
            # Jacobian for the i-th point:
            Ji, ri = compute_camera_jacobian_robust(K, R_cur, t_cur, points[i], measurements[i])
            # Add to full jacobian
            JTJ += dot(Ji.T, Ji)    # 6x2 * 2x6
            JTr += dot(Ji.T, ri)    # 6x2 * 2x1

        # Pick a step length
        while not converged:
            print '  damping=',damping
            # Apply Levenberg-Marquardt damping
            A = JTJ.copy()
            for i in range(6):
                A[i] *= (1. + damping)
    
            # Solve normal equations
            update = -solve(A, JTr)
            print '  ',update

            # Take gradient step
            R_next = dot(R_cur, SO3_exp(update[:3]))
            t_next = t_cur + update[3:]

            # Compute new cost
            cost_next = cost_robust(points, measurements, K, R_next, t_next)
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
                if damping > 1e+8:
                    print 'FAILED TO CONVERGE'
                    break

    return R_cur, t_cur




#
# Run tests
#

def run_with_synthetic_data():
    NUM_POINTS = 100
    MEASUREMENT_NOISE = .01

    seed(1211)

    R = eye(3)
    t = array([5., 1.,  5.])

    # Generate some points
    xs = randn(NUM_POINTS, 3)
    
    # Generate some measurements
    measurements = array([ dot(K, dot(R,x)+t) for x in xs ])
    measurements = measurements[:,:2] / measurements[:,2:]   # pinhole projection

    # Add some measurement noise
    measurements += randn(*measurements.shape) * MEASUREMENT_NOISE

    # Pick a starting point for optimization
    th = .3
    R_init = array([[ cos(th), -sin(th),  0. ],
                    [ sin(th),  cos(th),  0. ],
                    [ 0      ,  0.      , 1. ]])
    t_init = array([ 5.1, .7, 4. ])


    pt_data = hstack((xs,measurements))
    savetxt(open('data/100measurements_zero_noise.txt', 'w'), pt_data, fmt='%10f')
    
    P = hstack((R_init, t_init[:,newaxis]))
    savetxt(open('data/init_pose.txt', 'w'), P, fmt='%10f')

    # Optimize
    R_opt, t_opt = optimize_pose(xs, measurements, K, R_init, t_init)
    
    # Report
    print R_opt
    print t_opt


def run_with_loaded_data():
    import sys
    if len(sys.argv) != 3:
        print 'Usage: python optimize_camera.py MEASUREMENT_DATA.txt INITIAL_POSE.txt'
        sys.exit(-1)

    try:
        data1 = loadtxt(open(sys.argv[1]))
        xs = data1[:,:3]
        measurements = data1[:,3:]
    except:
        print 'Failed to load measurements from %s\nFormat is "x y z image_x image_y" for each row' % sys.argv[1]
        sys.exit(-1)

    try:
        data2 = loadtxt(open(sys.argv[2]))
        assert data2.shape == (3,4), 'Data in %s should be a 3x4 matrix' % sys.argv[2]
        R_init = data2[:,:3]
        t_init = data2[:,3]
    except:
        print 'Failed to load initial pose from %s\nFormat is a single 3x4 projection matrix' % sys.argv[2]
        sys.exit(-1)

    print 'Initial pose:'
    print hstack((R_init, t_init[:,newaxis]))

    R_opt, t_opt = optimize_pose(xs, measurements, K, R_init, t_init)
    
    print 'Polished pose:'
    print hstack((R_opt, t_opt[:,newaxis]))

if __name__ == '__main__':
    run_with_loaded_data()
    #unittest.main()


