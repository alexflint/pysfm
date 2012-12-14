import sys
import numpy as np
from numpy import *
from copy import deepcopy

import bundle
import sensor_model
import optimize
from algebra import *
from bundle_adjuster import BundleAdjuster
import draw_bundle
from lie import SO3

############################################################################
def rotation_xy(th):
    return np.array([[ np.cos(th), -np.sin(th), 0 ],
                     [ np.sin(th), np.cos(th),  0 ],
                     [ 0,          0,           1 ]])

############################################################################
def rotation_xz(th):
    return np.array([[ np.cos(th),  0,  -np.sin(th) ],
                     [ 0,           1,  0           ],
                     [ np.sin(th),  0,  np.cos(th), ]])

############################################################################
def create_test_bundle(noise=.1):
    NUM_CAMERAS        = 4
    NUM_POINTS         = 12
    POINT_CLOUD_RADIUS = 5.
    MISSING_FRAC       = .1     # fraction of measurements that are missing
    OUTLIER_FRAC       = 0      # fraction of measurements that are outliers
    OUTLIER_RANGE      = 10.    # bounds of uniform distribution from which outliers are sampled
    CAUCHY_PARAM       = .05    # determines the width of the robustifier

    # Setup some cameras
    np.random.seed(1111)  # for repeatability
    K = np.array([[  2.,  0., -.5  ],
                  [ .05,  3.,  .1  ],
                  [  0.,  0.,   1. ]])

    Rs = [ eye(3), rotation_xy(.3), rotation_xz(.4), rotation_xy(.1) ]
    ts = np.array([[  0.,  0.,  0. ],
                   [ -1,   0.,  1. ],
                   [  1.,  2.,  3. ],
                   [  0.,  0., -1. ]])

    # Sample 3D points
    pts = np.random.randn(NUM_POINTS, 3) * POINT_CLOUD_RADIUS
    pts[:,2] += 10   # ensure points are not close to focal plane

    # Compute ideal projections and add noise
    msm = np.array([[ bundle.project(K, R, t, pt) for pt in pts ]
                    for (R,t) in zip(Rs,ts) ])
    msm += np.random.randn(*msm.shape) * noise

    # Mark some measurements as missing
    np.random.seed(4309)  # for repeatability
    msm_mask = np.ones(msm.shape[:2], bool)
    nmissing = int(MISSING_FRAC * NUM_POINTS)
    for i in range(NUM_CAMERAS):
        missing_inds = np.random.permutation(NUM_POINTS)[:nmissing]
        msm_mask[i, missing_inds] = False

    # Generate some outliers by replacing measurements with random data
    np.random.seed(101)  # for repeatability
    outlier_mask = np.zeros((NUM_CAMERAS, NUM_POINTS), bool)
    noutliers = int(OUTLIER_FRAC * NUM_POINTS)
    for i in range(NUM_CAMERAS):
        outlier_inds = np.random.permutation(NUM_POINTS)[:noutliers]
        outlier_mask[i,outlier_inds] = True
        for j in outlier_inds:
            msm[i,j] = np.random.uniform(-OUTLIER_RANGE, OUTLIER_RANGE, 2)

    # Create the bundle
    b = bundle.Bundle.FromArrays(K, Rs, ts, pts, msm, msm_mask)

    # Attach robustifier
    b.sensor_model = sensor_model.CauchyModel(CAUCHY_PARAM)

    # Store this for visualization later
    b.outlier_mask = outlier_mask
    return b

############################################################################
def uniform_random(lo, hi, shape):
    return np.random.rand(*shape) * (hi-lo) + lo

############################################################################
def create_planar_bundle(noise=.1):
    NUM_CAMERAS        = 4
    NUM_POINTS         = 12
    POINT_CLOUD_RADIUS = 5.
    MISSING_FRAC       = 0      # fraction of measurements that are missing
    OUTLIER_FRAC       = 0      # fraction of measurements that are outliers
    OUTLIER_RANGE      = 10.    # bounds of uniform distribution from which outliers are sampled
    CAUCHY_PARAM       = .05    # determines the width of the robustifier

    # Setup some cameras
    np.random.seed(1111)  # for repeatability
    K = np.array([[  2.,  0., -.5  ],
                  [ .05,  3.,  .1  ],
                  [  0.,  0.,   1. ]])

    # Random rotations; zero translation
    Rs = [ SO3.exp(np.random.rand(3)*.1) for i in range(NUM_CAMERAS) ]
    ts = np.zeros((NUM_CAMERAS, 3))

    # Sample 3D points
    r = POINT_CLOUD_RADIUS
    pts = np.random.uniform(-r, r, (NUM_POINTS, 3))
    pts[:,2] += 10   # ensure points are not close to focal plane

    # Compute ideal projections and add noise
    msm = np.array([[ bundle.project(K, R, t, pt) for pt in pts ]
                    for (R,t) in zip(Rs,ts) ])
    msm += np.random.randn(*msm.shape) * noise

    # Mark some measurements as missing
    np.random.seed(4309)  # for repeatability
    msm_mask = np.ones(msm.shape[:2], bool)
    nmissing = int(MISSING_FRAC * NUM_POINTS)
    if nmissing > 0:
        for i in range(NUM_CAMERAS):
            missing_inds = np.random.permutation(NUM_POINTS)[:nmissing]
            msm_mask[i, missing_inds] = False

    # Generate some outliers by replacing measurements with random data
    np.random.seed(101)  # for repeatability
    outlier_mask = np.zeros((NUM_CAMERAS, NUM_POINTS), bool)
    noutliers = int(OUTLIER_FRAC * NUM_POINTS)
    if noutliers > 0:
        for i in range(NUM_CAMERAS):
            outlier_inds = np.random.permutation(NUM_POINTS)[:noutliers]
            outlier_mask[i,outlier_inds] = True
            for j in outlier_inds:
                msm[i,j] = np.random.uniform(-OUTLIER_RANGE, OUTLIER_RANGE, 2)

    # Create the bundle
    b = bundle.Bundle.FromArrays(K, Rs, ts, pts, msm, msm_mask)

    # Attach robustifier
    b.sensor_model = sensor_model.GaussianModel(.1)
    #b.sensor_model = sensor_model.CauchyModel(CAUCHY_PARAM)

    # Store this for visualization later
    b.outlier_mask = outlier_mask
    return b

############################################################################
def create_test_problem(noise=.05, initial_perturbation=.1):
    # Create the ground truth bundle
    #b_true = create_test_bundle(noise)
    b_true = create_planar_bundle(noise)

    # Set up the parameter mask
    # NOTE: we no longer need this because we can use
    # BundleAdjuster.optim_camera_ids instead (it is a list of the
    # camera IDs to optimize).
    #param_mask = np.array([ i>=6 for i in range(b_true.num_params()) ])
    #param_mask[9] = False
    #assert len(b_init.optim_camera_ids) == len(

    # Pick a starting point
    np.random.seed(1888)
    initialization_offset = np.random.randn(b_true.num_params()) * initial_perturbation
    b_init = deepcopy(b_true).perturb(initialization_offset)

    # Make all translations norm 1
    #for camera in b_init.cameras:
    #    camera.t /= np.linalg.norm(camera.t)

    # Replace the initial 3D point estimates with their triangulations
    # given the initial camera parameters
    b_init.triangulate_all()

    return b_true, b_init


############################################################################
# This version uses explicit LM updates
def test_optimize_raw_lm():
    MAX_STEPS = 25

    print '***\nOPTIMIZING WITH RAW LEVENBERG-MARQUARDT\n***'
    b_true, b_init = create_test_problem()
    # Note that we no longer have a parameter mask because that work
    # is done by setting BundleAdjuster.optim_cam_ids

    # Begin optimizing
    bcur = deepcopy(b_init)
    costs = [ bcur.complete_cost() ]
    lm = optimize.LevenbergMarquardt()
    while not lm.converged and lm.num_steps < MAX_STEPS:
        print 'Step %d: cost= %f, damping=%f' % (lm.num_steps, bcur.complete_cost(), lm.damp)
        rcur = bcur.residuals()
        Jcur = bcur.Jresiduals()[:,param_mask]
        while not lm.converged:
            delta = lm.next_update(rcur, Jcur)
            bnext = deepcopy(bcur).perturb(delta, param_mask)
            rnext = bnext.residuals()
            if np.dot(rnext,rnext) < np.dot(rcur,rcur):
                lm.accept_update(rnext)
                costs.append(bnext.complete_cost())
                bcur = bnext
                break
            else:
                lm.reject_update()

    if lm.converged:
        print 'Converged after %d steps' % lm.num_steps
    else:
        print 'Failed to converge after %d steps' % lm.num_steps

    report_bundle(b_init, 'Initial')
    report_bundle(bcur, 'Estimated')
    report_bundle(b_true, 'True')

    print '\nError in ts:'
    print abs(np.array(b_true.ts) - bcur.ts)
    print 'Error in Rs:'
    print abs(asarray(b_true.Rs) - bcur.Rs)
    print 'Error in points:'
    print sum(square(abs(array(b_true.pts) - bcur.pts)), axis=1)

    #print '\nEvolution of cost function:'
    #print '  ',costs

    print '\nCost (initial -> estimated -> true)'
    print '  %f -> %f -> %f' % (b_init.complete_cost(), bcur.complete_cost(), b_true.complete_cost())

############################################################################
def report_bundle(bundle, name):
    bundle.make_relative_to_first_camera()
    print '\n',name,' cameras:'
    for camera in bundle.cameras:
        print camera
    print name,'structure:'
    print np.array(bundle.reconstruction)
    #print 'Initial reproj errors:'
    #print b_init.reproj_errors()
    #print 'Initial residuals:'
    #print b_init.residuals()
    print name, 'cost:'
    print bundle.complete_cost()

############################################################################
# This version uses the schur complement
def test_optimize_fast():
    b_true, b_init = create_test_problem(noise=0)

    ba = BundleAdjuster(b_init)
    ba.optimize(max_steps=50)

    report_bundle(b_init, 'Initial')
    report_bundle(ba.bundle, 'Estimated')
    report_bundle(b_true, 'True')

    #print '\nError in ts:'
    #print abs(np.array(b_true.ts) - bcur.ts)
    #print 'Error in Rs:'
    #print abs(asarray(b_true.Rs) - bcur.Rs)
    #print 'Error in points:'
    #print sum(square(abs(array(b_true.pts) - bcur.pts)), axis=1)

    #print '\nOffset in Rs:'
    #for Rtrue,Rinit in zip(b_true.Rs, b_init.Rs):
    #    print dots(Rtrue, Rinit.T)

    print '\nCost (initial -> estimated -> true)'
    print '  %f -> %f -> %f' % (b_init.complete_cost(), ba.bundle.complete_cost(), b_true.complete_cost())

    draw_bundle.output_views(b_init, 'out/init.pdf')
    draw_bundle.output_views(b_true, 'out/true.pdf')
    draw_bundle.output_views(ba.bundle, 'out/estimated.pdf')

############################################################################
if __name__ == '__main__':
    # disable scientific notation
    np.set_printoptions(suppress=True,
                        precision=4)

    #test_optimize()
    #test_optimize2()
    test_optimize_fast()
