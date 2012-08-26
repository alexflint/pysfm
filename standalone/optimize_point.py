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

# Compute the sum of Cauchy-robustified reprojection errors
def cost_robust(measurements, Rs, ts, x):
    error = 0.
    for i in range(len(measurements)):
        projection = dot(K, dot(Rs[i], x) + ts[i])
        projection = projection[:2] / projection[2]
        reproj_error = projection - measurements[i]
        error += cauchy_cost_from_reprojection_error(reproj_error)  # sum of squared differences
    return error

# Compute the reprojection error of x given a camera (R,t) with
# intrinsics K. Also compute the Jacobian with respect to x.
# Parameters:
#   R -- 3x3 camera rotation
#   t -- 3x1 camera translation
#   x -- 3x1 point
#   measurement -- 2x1 measured position of x in camera
#
# Let f(K, R, t, x) = pr(K * (R * x + t))
#   where pr(x,y,z) = [ x/z, y/z ]
#
# Then the Jacobian of f with respect to x is:
#   Jf_t = Jpr(K * (R * x + t)) * K * R
# where Jpr is the Jacobian of g(x,y,z) = cauchy(x/z, y/z)
def compute_point_jacobian_robust(R, t, x, measurement):
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
    # Robustify
    residual, Jresidual = cauchy_residual_from_reprojection_error(prediction - measurement)
    # Jacobian of the projection with respect to 3D point
    J = dots4(Jresidual, Jpr, K, R)

    return J, residual

#
# Optimization
#

# Given:
#   measurements -- a list of 2-vectors representing measurements of those points
#   Rs           -- a list of 3x3 rotations representing camera rotations
#   ts           -- a list of 3-vectors representing camera translations
#   x_init       -- an initial guess for the 3D point 
# Find a point X minimizing:
#   cost_robust(measurements, Rs, ts, X)
def optimize_point(measurements, Rs, ts, x_init):
    num_steps = 0
    x_cur = x_init.copy()
    cost_cur = cost_robust(measurements, Rs, ts, x_cur)
    converged = False
    damping = INIT_DAMPING
    while num_steps < MAX_STEPS and damping < 1e+8 and not converged:
        # Print status
        print 'Step %d: cost=%f' % (num_steps, cost_cur)

        # Compute J^T * J and J^T * r
        JTJ = zeros((3,3))  # 3x3
        JTr = zeros(3)      # 3x1
        for i in range(len(Rs)):
            # Jacobian for the i-th point:
            Ji, ri = compute_point_jacobian_robust(Rs[i], ts[i], x_cur, measurements[i])
            # Add to full jacobian
            JTJ += dot(Ji.T, Ji)    # 3x2 * 2x3 -> 3x3
            JTr += dot(Ji.T, ri)    # 3x2 * 2x1 -> 3x1

        # Pick a step length
        while not converged:
            # Check for failure
            if damping > 1e+8:
                print 'FAILED TO CONVERGE'
                break

            # Apply Levenberg-Marquardt damping
            A = JTJ.copy()
            for i in range(3):
                A[i] *= (1. + damping)
    
            # Solve normal equations: 3x3
            try:
                update = -solve(A, JTr)
            except LinAlgError:
                # Badly conditioned: increase damping and try again
                damping *= 10.
                continue

            # Take gradient step
            x_next = x_cur + update

            # Compute new cost
            cost_next = cost_robust(measurements, Rs, ts, x_next)
            if cost_next < cost_cur:
                # Cost decreased: accept the update
                if cost_cur - cost_next < CONVERGENCE_THRESH or damping < 1e-8:
                    converged = True

                x_cur = x_next
                cost_cur = cost_next
                damping *= .1
                num_steps += 1
                break

            else:
                # Cost increased: reject the udpate, try another step length
                damping *= 10.

    return x_cur

####################################################################################
# End of main optimization stuff
####################################################################################












# This function is identical to optimize_point except that it
# initializes optimization with a point triangulated by least squares
# over the algebraic error.
def optimize_point_using_lsq_initialization(measurements, Rs, ts):
    x_init = triangulate_algebraic_lsq(K, Rs, ts, measurements)
    return optimize_point(measurements, Rs, ts, x_init), x_init

# This function is identical to optimize_point except that it
# initializes optimization with a point estimated using RANSAC over
# the triangulation error.
def optimize_point_using_ransac_initialization(measurements, Rs, ts):
    measurements = asarray(measurements)
    Rs = asarray(Rs)
    ts = asarray(ts)

    # RANSAC iterations
    best_cost = inf
    for i in range(100):
        # Sample 2 measurements
        sample = permutation(len(measurements))[:2]
        # Compute minimal solution
        x_hyp = triangulate_algebraic_lsq(K,
                                          Rs[sample],
                                          ts[sample],
                                          measurements[sample])
        # Compute robust cost
        cur_cost = cost_robust(measurements, Rs, ts, x_hyp)
        if cur_cost < best_cost:
            best_cost = cur_cost
            x_init = x_hyp

    return optimize_point(measurements, Rs, ts, x_init), x_init

#
# Run tests
#

def run_with_synthetic_data():
    NUM_CAMERAS = 100
    MEASUREMENT_NOISE = .01
    NUM_OUTLIERS = 10

    seed(4875)  # repeatability

    # Pick a true point location
    x_true = array([4., .5, -20.])  # far from origin so that it is never close to a focal plane

    # Generate some poses
    Rs = array([ SO3_exp(randn(3) * .5) for i in range(NUM_CAMERAS) ])
    ts = array([ randn(3) * 4 for i in range(NUM_CAMERAS) ])
    
    # Generate some measurements
    measurements = array([ dot(K, dot(R,x_true)+t) for R,t in zip(Rs,ts) ])
    measurements = measurements[:,:2] / measurements[:,2:]   # pinhole projection

    # Add some measurement noise
    measurements += randn(*measurements.shape) * MEASUREMENT_NOISE

    # Add some outliers
    outliers = permutation(len(measurements))[:NUM_OUTLIERS]
    measurements[outliers] = randn(len(outliers), 2) * 3.

    # Save to file
    pose_data = hstack(( vstack(Rs), ts.flatten()[:,newaxis] ))
    savetxt(open('point_estimation_data/100poses.txt', 'w'), pose_data, fmt='%10f')    
    savetxt(open('point_estimation_data/100measurements_with_outliers.txt', 'w'), measurements, fmt='%10f')

    # Optimize
    #x_opt,x_init = optimize_point_using_lsq_initialization(measurements, Rs, ts)
    x_opt,x_init = optimize_point_using_ransac_initialization(measurements, Rs, ts)
    
    # Report
    print '\nTrue point:'
    print x_true
    print '\nInitial point (from RANSAC):'
    print x_init
    print '\nFinal polished point:'
    print x_opt


def run_with_data_from_file():
    import sys
    if len(sys.argv) != 3:
        print 'Usage: python %s POSES.txt MEASUREMENTS.txt' % sys.argv[0]
        sys.exit(-1)

    try:
        posedata = loadtxt(open(sys.argv[1]))
        assert posedata.shape[1] == 4  # check that there are 4 cols
        Rs = posedata[:,:3].reshape((-1, 3, 3))
        ts = posedata[:,3].reshape((-1, 3))
        assert len(Rs) == len(ts)
    except Exception as ex:
        print 'Failed to load poses from %s\nFormat is 3 rows per camera, each with 4 entries' % sys.argv[1]
        raise ex
        sys.exit(-1)

    try:
        measurements = loadtxt(open(sys.argv[2]))
        assert measurements.shape[1] == 2
        assert len(measurements) == len(Rs)
    except:
        print 'Failed to load measurements from %s\nFormat is rows of "<x> <y>"' % sys.argv[2]
        sys.exit(-1)

    x_opt, x_init = optimize_point_using_ransac_initialization(measurements, Rs, ts)

    # Report
    print '\nInitial point (from RANSAC):'
    print x_init
    print '\nFinal polished point:'
    print x_opt



if __name__ == '__main__':
    #run_with_synthetic_data()
    run_with_data_from_file()
