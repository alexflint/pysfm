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

############################################################################
def rotation_xy(th):
    return np.array([[ np.cos(th), -np.sin(th), 0 ],
                     [ np.sin(th), np.cos(th),  0 ],
                     [ 0,          0,           1 ]])

def rotation_xz(th):
    return np.array([[ np.cos(th),  0,  -np.sin(th) ],
                     [ 0,           1,  0           ],
                     [ np.sin(th),  0,  np.cos(th), ]])

def create_test_bundle(noise=.1):
    NUM_CAMERAS        = 4
    NUM_POINTS         = 50
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
def create_test_problem(noise=.05, initial_pert=.1):
    # Create the ground truth bundle
    b_true = create_test_bundle(noise)

    # Set up the parameter mask
    param_mask = np.array([ i>=6 for i in range(b_true.num_params()) ])
    param_mask[9] = False

    # Pick a starting point
    np.random.seed(1888)
    pert = np.zeros(b_true.num_params())
    pert[param_mask] = np.random.randn(np.sum(param_mask)) * initial_pert
    b_init = deepcopy(b_true).perturb(pert)

    # Replace the initial 3D point estimates with their triangulations
    # given the initial camera parameters
    for track in b_init.tracks:
        track.reconstruction = b_init.triangulate(track)

    return b_true, b_init, param_mask

############################################################################
def report_bundle(bundle, name):
    print '\n',name,'ts:'
    print array(bundle.ts)
    print 'Initial Rs:'
    print array(bundle.Rs)
    print name,'points:'
    print np.array(bundle.pts)
    #print 'Initial reproj errors:'
    #print b_init.reproj_errors()
    #print 'Initial residuals:'
    #print b_init.residuals()
    print name, 'cost:'
    print bundle.cost()


############################################################################
# This version uses explicit LM updates
def test_optimize_raw_lm():
    MAX_STEPS = 25

    print '***\nOPTIMIZING WITH RAW LEVENBERG-MARQUARDT\n***'
    b_true, b_init, param_mask = create_test_problem()

    # Begin optimizing
    bcur = deepcopy(b_init)
    costs = [ bcur.cost() ]
    lm = optimize.LevenbergMarquardt()
    while not lm.converged and lm.num_steps < MAX_STEPS:
        print 'Step %d: cost= %f, damping=%f' % (lm.num_steps, bcur.cost(), lm.damp)
        rcur = bcur.residuals()
        Jcur = bcur.Jresiduals()[:,param_mask]
        while not lm.converged:
            delta = lm.next_update(rcur, Jcur)
            bnext = deepcopy(bcur).perturb(delta, param_mask)
            rnext = bnext.residuals()
            if np.dot(rnext,rnext) < np.dot(rcur,rcur):
                lm.accept_update(rnext)
                costs.append(bnext.cost())
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
    print '  %f -> %f -> %f' % (b_init.cost(), bcur.cost(), b_true.cost())


############################################################################
# This version uses the schur complement
def test_optimize_fast():
    print '***\nOPTIMIZING WITH FAST SCHUR\n***'
    b_true, b_init, param_mask = create_test_problem()

    ba = BundleAdjuster(b_init)
    ba.optimize(param_mask, max_steps=25)

    #report_bundle(b_init, 'Initial')
    #report_bundle(bcur, 'Estimated')
    #report_bundle(b_true, 'True')

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
    print '  %f -> %f -> %f' % (b_init.cost(), ba.bundle.cost(), b_true.cost())

    draw_bundle.output_views(b_init, 'out/init.pdf')
    draw_bundle.output_views(b_true, 'out/true.pdf')
    draw_bundle.output_views(ba.bundle, 'out/estimated.pdf')

############################################################################
if __name__ == '__main__':
    #test_optimize()
    #test_optimize2()
    test_optimize_fast()
