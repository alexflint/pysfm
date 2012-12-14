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
def compute_scale_jacobian_robust(R, t, x, measurement):
    R = asarray(R)
    t = asarray(t)
    x = asarray(x)
    measurement = asarray(measurement)

    # projection: p = K * (R*x + t)
    p = dot(K, dot(R, x) + t)

    # jacobian of projection with respect to p
    Jpr = array([[ 1./p[2],  0.,       -p[0] / (p[2]*p[2]) ],
                 [ 0.,       1./p[2],  -p[1] / (p[2]*p[2]) ]])

    # jacobian of projection with respect to scale
    Js = dot(Jpr, t)

    # divide by z
    prediction = p[:2] / p[2]
    reproj_error = prediction - measurement

    # robustify
    residual, Jresidual = cauchy_residual_from_reprojection_error(reproj_error)
    Jpr = dots(Jresidual, Js)

    return Jpr, residual

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
def optimize_pose(points, measurements, R_fixed, t_init):
    num_steps = 0
    t_cur = t_init
    cost_cur = cost_robust(points, measurements, R_fixed, t_cur)
    converged = False
    damping = INIT_DAMPING

    while num_steps < MAX_STEPS and damping < 1e+8 and not converged:
        # Print status
        print 'Step %d: cost=%f, damping=%f' % (num_steps, cost_cur, damping)

        # Compute gradient and residual
        gradient = 0.
        residual = 0.
        for i in range(len(points)):
            # Jacobian for the i-th point:
            Ji, ri = compute_scale_jacobian_robust(R_fixed, t_cur, points[i], measurements[i])
            gradient += dot(Ji, Ji)
            residual += dot(Ji, ri)

        # If |gradient| < eps then we're done
        if abs(gradient) < 1e-8:
            converged = True
            break

        # Take gradient step
        while not converged and damping < 1e+8:
            t_next = t_cur * (1. - residual / (gradient * (1. + damping)))

            # Compute new cost
            cost_next = cost_robust(points, measurements, R_fixed, t_next)
            if cost_next < cost_cur:
                # Cost decreased: accept the update
                if cost_cur - cost_next < CONVERGENCE_THRESH:
                    converged = True

                if damping > 1e-15:
                    damping /= 10.

                t_cur = t_next
                cost_cur = cost_next
                num_steps += 1
                break

            else:
                # Cost increased: reject the udpate, try another step length
                damping *= 10.

    return t_cur

####################################################################################
# End of main optimization stuff
####################################################################################












def run_with_synthetic_data():
    NUM_POINTS = 100
    MEASUREMENT_NOISE = .1
    NUM_OUTLIERS = 0

    seed(1211)  # repeatability

    R_true = eye(3)
    t_true = array([5., 1.,  5.])

    # Generate some points
    xs = randn(NUM_POINTS, 3)
    
    # Generate some measurements
    measurements = array([ dot(K, dot(R_true,x)+t_true) for x in xs ])
    measurements = measurements[:,:2] / measurements[:,2:]   # pinhole projection

    # Add some measurement noise
    measurements += randn(*measurements.shape) * MEASUREMENT_NOISE

    # Add some outliers
    outliers = permutation(len(measurements))[:NUM_OUTLIERS]
    measurements[outliers] = randn(len(outliers), 2) * 3.

    # Pick a starting point for optimization
    t_init = t_true * 2.5

    # Save to file
    pt_data = hstack((xs,measurements))
    savetxt(open('scale_estimation_data/100_measurements_tenpercent_noise.txt', 'w'), pt_data, fmt='%10f')
    
    P = hstack((R_true, t_init[:,newaxis]))
    savetxt(open('scale_estimation_data/init_pose.txt', 'w'), P, fmt='%10f')

    # Optimize
    t_opt = optimize_pose(xs, measurements, R_true, t_init)
    
    # Report
    print '\nTrue translation:'
    print t_true
    print '\nInitial translation:'
    print t_init
    print '\nFinal polished translation:'
    print t_opt


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

    t_opt = optimize_pose(xs, measurements, R_init, t_init)

    # Report
    print '\nInitial translation:'
    print t_init
    print '\nFinal polished translation:'
    print t_opt


if __name__ == '__main__':
    #run_with_synthetic_data()
    run_with_data_from_file()
