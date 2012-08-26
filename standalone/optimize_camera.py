# Import python's numerics library
from numpy import *
from numpy.linalg import *
from numpy.random import *

from helpers import *

# Notes on python numerics:
# A[i,:]   -- get the i-th row of A
# A[:,i]   -- get the i-th column of A
# A[:5]    -- get the first five rows of of A
# A[:,:2]  -- get the first two columns of of A

# Optimization parameters
MAX_STEPS = 100
CONVERGENCE_THRESH = 1e-5   # convergence detected when improvement less than this
INIT_DAMPING = 10.
         
# Camera calibration
K = array([[ 1.,   0.,   0. ],
           [ 0.,   1.,   0. ],
           [ 0.,   0.,   1. ]])

# Compute the reprojection error of x given a camera (R,t) with
# intrinsics K. Also compute the Jacobian with respect to R and t.
#   R -- 3x3 camera rotation
#   t -- 3x1 camera translation
#   x -- 3x1 point
#   measurement -- 2x1 measured position of x in camera
#
# Let f(R, t, x) = pr(K * (R * x + t))
#   where pr(x,y,z) = [ x/z, y/z ]
#
# Then the Jacobian of f with respect to t is:
#   Jf_t = Jpr(K * (R * x + t)) * K
# and the Jacobian of f with respect to R is:
#   Jf_R = Jpr(K * (R * x + t)) * K
# where Jpr is the Jacobian of (x,y,z) -> (x/z, y/z)
def compute_camera_jacobian_robust(R, t, x, measurement):
    R = asarray(R)
    t = asarray(t)
    x = asarray(x)
    measurement = asarray(measurement)

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
    Jpr = dots3(Jresidual, Jpr, K)

    # Jacobian of the projection with respect to rotation
    JR = dots3(Jpr, R, skew(-x))
    # Jacobian of the projection with respect to transation
    Jt = Jpr
    # Concatenate
    J = hstack((JR, Jt))

    return J, residual

# Compute the sum of Cauchy-robustified reprojection errors
def cost_robust(points, measurements, R, t):
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
#   points       -- a list of 3-vectors representing points in space
#   measurements -- a list of 2-vectors representing measurements of those points
#   R_init       -- initial camera rotation, 3x3 matrix
#   t_init       -- initial camera translation, 3-vector
# Find a new rotation and translation minimizing:
#   cost_robust(points, measurements, K, R, t)
def optimize_pose(points, measurements, R_init, t_init):
    num_steps = 0
    R_cur = R_init
    t_cur = t_init
    cost_cur = cost_robust(points, measurements, R_cur, t_cur)
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
            Ji, ri = compute_camera_jacobian_robust(R_cur, t_cur, points[i], measurements[i])
            # Add to full jacobian
            JTJ += dot(Ji.T, Ji)    # 6x2 * 2x6
            JTr += dot(Ji.T, ri)    # 6x2 * 2x1

        # Pick a step length
        while not converged:
            # Apply Levenberg-Marquardt damping
            A = JTJ.copy()
            for i in range(6):
                A[i] *= (1. + damping)
    
            # Solve normal equations
            update = -solve(A, JTr)

            # Take gradient step
            R_next = dot(R_cur, SO3_exp(update[:3]))
            t_next = t_cur + update[3:]

            # Compute new cost
            cost_next = cost_robust(points, measurements, R_next, t_next)
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

####################################################################################
# End of main optimization stuff
####################################################################################












def run_with_synthetic_data():
    NUM_POINTS = 100
    MEASUREMENT_NOISE = .1
    NUM_OUTLIERS = 10

    seed(1211)  # repeatability

    R = eye(3)
    t = array([5., 1.,  5.])

    # Generate some points
    xs = randn(NUM_POINTS, 3)
    
    # Generate some measurements
    measurements = array([ dot(K, dot(R,x)+t) for x in xs ])
    measurements = measurements[:,:2] / measurements[:,2:]   # pinhole projection

    # Add some measurement noise
    measurements += randn(*measurements.shape) * MEASUREMENT_NOISE

    # Add some outliers
    outliers = permutation(len(measurements))[:NUM_OUTLIERS]
    measurements[outliers] = randn(len(outliers), 2) * 3.

    # Pick a starting point for optimization
    th = .3
    R_init = array([[ cos(th), -sin(th),  0. ],
                    [ sin(th),  cos(th),  0. ],
                    [ 0      ,  0.      , 1. ]])
    t_init = array([ 5.1, .7, 4. ])

    # Save to file
    pt_data = hstack((xs,measurements))
    savetxt(open('pose_estimation_data/100measurements_with_outliers.txt', 'w'), pt_data, fmt='%10f')
    
    P = hstack((R_init, t_init[:,newaxis]))
    savetxt(open('pose_estimation_data/init_pose.txt', 'w'), P, fmt='%10f')

    # Optimize
    R_opt, t_opt = optimize_pose(xs, measurements, R_init, t_init)
    
    # Report
    P_opt = hstack((R_opt, t_opt[:, newaxis]))
    P_init = hstack((R_init, t_init[:, newaxis]))
    print '\nInitial pose:'
    print P_init
    print '\nFinal polished pose:'
    print P_opt


def run_with_data_from_file():
    import sys
    if len(sys.argv) != 3:
        print 'Usage: python %s MEASUREMENTS.txt INITIAL_POSE.txt' % sys.argv[0]
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

    R_opt, t_opt = optimize_pose(xs, measurements, R_init, t_init)

    # Report
    P_opt = hstack((R_opt, t_opt[:, newaxis]))
    P_init = hstack((R_init, t_init[:, newaxis]))
    print '\nInitial pose:'
    print P_init
    print '\nFinal polished pose:'
    print P_opt



if __name__ == '__main__':
    #run_with_synthetic_data()
    run_with_data_from_file()
