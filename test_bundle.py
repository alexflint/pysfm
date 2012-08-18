import sys
import numpy as np
from numpy import *

import bundle
import robustifier
import optimize
import lie
from algebra import *
import finite_differences
import schur
import triangulate

############################################################################
def rotation_xy(th):
    return np.array([[ np.cos(th), -np.sin(th), 0 ],
                     [ np.sin(th), np.cos(th),  0 ],
                     [ 0,          0,           1 ]])

def rotation_xz(th):
    return np.array([[ np.cos(th),  0,  -np.sin(th) ],
                     [ 0,           1,  0           ],
                     [ np.sin(th),  0,  np.cos(th), ]])

def create_test_bundle(noise=.5):
    NUM_CAMERAS = 4
    NUM_POINTS = 50
    POINT_CLOUD_RADIUS = 5.
    MISSING_FRAC = .1     # 10% of measurements are missing
    OUTLIER_FRAC = .1     # 10% of measurements are total outliers
    np.random.seed(1111)  # repeatability

    # Create a bundle
    b = bundle.Bundle(NUM_CAMERAS, NUM_POINTS)
    b.K = np.array([[ 2.,  0., -.5  ],
                    [.05,  3.,  .1  ],
                    [ 0.,  0.,   1. ]])

    b.Rs = [ eye(3), rotation_xy(.3), rotation_xz(.4), rotation_xy(.1) ]
    b.ts = np.array([[ 0.,  0.,  0.],
                     [ -1,  0.,  1.],
                     [ 1.,  2.,  3.],
                     [ 0.,  0.,  -1.]])
    b.pts = np.random.randn(NUM_POINTS, 3) * POINT_CLOUD_RADIUS
    b.pts[:,2] += 10   # ensure points are not close to focal plane

    # Attach robustifier
    CR = robustifier.CauchyRobustifier(.2)
    b.robustifier = CR.cost
    b.Jrobustifier = CR.Jcost

    # Populate measurements with ideal projections plus measurement noise
    b.msm = b.predictions().reshape(b.msm.shape) + np.random.randn(*b.msm.shape)*noise

    # Mark some measurements as missing
    nmissing = int(MISSING_FRAC * NUM_POINTS)
    for i in range(NUM_CAMERAS):
        b.msm_mask[i, np.random.randint(0, NUM_POINTS, nmissing)] = False

    # Generate some outliers by replacing measurements with random data
    noutliers = int(OUTLIER_FRAC * NUM_POINTS)
    for i in range(NUM_CAMERAS):
        outlier_inds = np.unique(np.random.randint(0, NUM_POINTS, nmissing))
        for j in outlier_inds:
            b.msm[i,j] = np.random.rand(2) * 10.

    return b

def create_test_problem(noise=.05, initial_pert=.1):
    # Create the ground truth bundle
    b_true = create_test_bundle(noise)

    # Set up the parameter mask
    param_mask = np.array([ i>=6 for i in range(b_true.nparams) ])
    param_mask[9] = False

    # Pick a starting point
    np.random.seed(1888)
    pert = np.zeros(b_true.nparams)
    pert[param_mask] = np.random.randn(np.sum(param_mask)) * initial_pert
    b_init = b_true.copy_and_update(pert)

    # Replace the initial 3D point estimates with their triangulations
    # given the initial camera parameters
    for j in range(b_init.npts):
        b_init.pts[j] = triangulate.triangulate(b_init.K, b_init.Rs, b_init.ts, b_init.msm[:,j])

    return b_true, b_init, param_mask

############################################################################
def check_projection_jacobians():
    K0 = np.array([[ 2., 0., -1.5 ],
                   [ .1, 3., .8 ],
                   [ 0., 0., 1. ]])
    R0 = rotation_xy(1.1)
    t0 = np.array([4., 2., -1.])
    x0 = np.array([ -1., 5., 2. ])

    print '\nChecking Jproject_t'
    finite_differences.check_jacobian(lambda t: bundle.project(K0, R0, t, x0),
                                      bundle.Jproject_t(K0, R0, t0, x0),
                                      t0)

    print '\nChecking Jproject_x'
    finite_differences.check_jacobian(lambda x: bundle.project(K0, R0, t0, x),
                                      bundle.Jproject_x(K0, R0, t0, x0),
                                      x0)

    print '\nChecking Jproject_R'
    finite_differences.check_jacobian(lambda m: bundle.project2(K0, R0, m, t0, x0),
                                      bundle.Jproject_R(K0, R0, t0, x0),
                                      np.zeros(3))


    print '\nChecking Cauchy robustifier'
    cr = robustifier.CauchyRobustifier(2.)
    finite_differences.check_jacobian(cr.cost, cr.Jcost, [1., 2.])
    finite_differences.check_jacobian(cr.cost, cr.Jcost, np.ones(2) * 1e-8)
    finite_differences.check_jacobian(cr.cost, cr.Jcost, zeros(2))

############################################################################
def check_bundle_jacobians():
    print '\nChecking bundle jacobians'
    b_init = create_test_bundle()
    return finite_differences.check_jacobian(
        lambda v: b_init.copy_and_update(v).residuals(),
        b_init.Jresiduals(),
        np.zeros(b_init.nparams))


############################################################################
def check_schur():
    b = create_test_bundle()

    r = b.residuals()
    J = b.Jresiduals();
    JTJ = dots(J.T, J)
    JTr = dots(J.T, r)

    optimize.apply_lm_damping_inplace(JTJ, 1.)
    Aslow, bslow = schur.get_schur_complement(JTJ, JTr, 6*b.ncameras)

    ba = bundle.BundleAdjuster(b)
    ba.prepare_schur_complement(1.)
    A,b = ba.compute_schur_complement()

    err = np.sum(np.abs(A-Aslow))
    print 'Error in Schur complement LHS: '+str(err)

    err = np.sum(np.abs(b-bslow))
    print 'Error in Schur complement RHS: '+str(err)

############################################################################
def check_schur_backsub():
    b = create_test_bundle()

    r = b.residuals()
    J = b.Jresiduals();
    JTJ = dots(J.T, J)
    JTr = dots(J.T, r)

    damping = 2.
    optimize.apply_lm_damping_inplace(JTJ, damping)
    delta_slow = np.linalg.solve(JTJ, JTr);

    ba = bundle.BundleAdjuster(b)
    ba.prepare_schur_complement(damping)
    A,b = ba.compute_schur_complement()
    dC = np.linalg.solve(A, b)
    dP = ba.backsubstitute(dC)

    dC_err = np.max(np.abs(dC - delta_slow[:len(dC)]))
    dP_err = np.max(np.abs(dP - delta_slow[len(dC):]))

    print 'Error in dP (Schur solution): '+str(dC_err)
    print 'Error in dC (Schur solution): '+str(dP_err)

############################################################################
def check_schur_update():
    b = create_test_bundle()
    damping = 2.

    r = b.residuals()
    J = b.Jresiduals();
    JTJ = dots(J.T, J)
    JTr = dots(J.T, r)

    optimize.apply_lm_damping_inplace(JTJ, damping)
    delta_slow = -np.linalg.solve(JTJ, JTr);

    ba = bundle.BundleAdjuster(b)
    delta = ba.compute_update(b, damping)

    delta_err = np.max(np.abs(delta - delta_slow))

    print 'Error in delta (Schur solution): '+str(delta_err)

############################################################################
# Optimize bundle parameters
def test_optimize():
    b = create_test_bundle()
    bundle_f = lambda x: replace_bundle_params(b, x).residuals()
    bundle_Jf = lambda x: replace_bundle_params(b, x).Jresiduals()

    # pick a starting point some distance from true params
    x0 = all_params(b)
    np.random.seed(1)
    x0[4:] += np.random.randn(len(x0)-4) * 1e-1

    opt = optimize.LevenbergMarquardtOptimizer(bundle_f, bundle_Jf);
    opt.param_mask = np.array([ i>3 for i in range(len(x0)) ])
    opt.optimize(x0)

    print 'True ts:'
    print b.ts
    print 'True Rs:'
    print b.Rs
    print 'True points:'
    print b.pts
    print 'Measurements:'
    print b.msm
    print 'Predictions:'
    print b.predictions()

    est = replace_bundle_params(b, opt.xcur)
    print 'Estimated ts:'
    print est.ts
    print 'Estimated Rs:'
    print est.Rs
    print 'Estimated points:'
    print est.pts
    print 'Predictions:'
    print est.predictions()

############################################################################
# This version uses a more efficient way of applying updates etc
def test_optimize2():
    print '***\nOPTIMIZING WITH RAW LEVENBERG-MARQUARDT\n***'
    b_true, b_init, param_mask = create_test_problem()

    # Begin optimizing
    bcur = b_init.copy()
    costs = [ bcur.cost() ]
    lm = optimize.LevenbergMarquardt()
    while not lm.converged and lm.num_steps < 100:
        print 'Step %d: cost= %f, damping=%f' % (lm.num_steps, bcur.cost(), lm.damp)
        rcur = bcur.residuals()
        Jcur = bcur.Jresiduals()[:,param_mask]
        while not lm.converged:
            delta = lm.next_update(rcur, Jcur)
            bnext = bcur.copy_and_update(delta, param_mask)
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

    # Report
    print '\nInitial ts:'
    print array(b_init.ts)
    print 'Initial Rs:'
    print array(b_init.Rs)
    print 'Initial points:'
    print np.array(b_init.pts)
    #print 'Initial reproj errors:'
    #print b_init.reproj_errors()
    #print 'Initial residuals:'
    #print b_init.residuals()
    print 'Initial cost:'
    print b_init.cost()

    print '\nEstimated ts:'
    print array(bcur.ts)
    print 'Estimated Rs:'
    print array(bcur.Rs)
    print 'Estimated points:'
    print np.array(bcur.pts)
    #print 'Estimated reproj errors:'
    #print bcur.reproj_errors()
    print 'Estimated cost:'
    print bcur.cost()

    print '\nTrue ts:'
    print array(b_true.ts)
    print 'True Rs:'
    print array(b_true.Rs)
    print 'True points:'
    print np.array(b_true.pts)
    #print 'True reproj errors:'
    #print b_true.reproj_errors()
    print 'True cost:'
    print b_true.cost()

    

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
def test_optimize3():
    print '***\nOPTIMIZING WITH FAST SCHUR\n***'
    b_true, b_init, param_mask = create_test_problem()

    # Begin optimizing
    bcur = b_init.copy()
    num_steps = 0
    damping = 100.
    converged = False
    costs = [ bcur.cost() ]
    while not converged and num_steps < 100:
        num_steps += 1
        cur_cost = bcur.cost()
        print 'Step %d: cost=%f, damping=%f' % (num_steps, cur_cost, damping)

        while not converged:
            ba = bundle.BundleAdjuster(bcur)
            try:
                delta = ba.compute_update(bcur, damping, param_mask)
            except np.linalg.LinAlgError:
                # Matrix was singular: increase damping
                damping *= 10.
                converged = damping > 1e+8
                continue

            bnext = bcur.copy_and_update(delta, param_mask) ##
            next_cost = bnext.cost()

            if next_cost < cur_cost:
                bcur = bnext
                damping *= .1
                costs.append(next_cost)
                converged = abs(cur_cost - next_cost) < 1e-8
                break
            else:
                damping *= 10.
                converged = damping > 1e+8

    if converged:
        print 'Converged after %d steps' % num_steps
    else:
        print 'Failed to converge after %d steps' % num_steps

    print '\nInitial ts:'
    print np.array(b_init.ts)
    print 'Initial Rs:'
    print np.array(b_init.Rs)
    print 'Initial points:'
    print np.array(b_init.pts)
    print 'Initial cost:'
    print b_init.cost()

    print '\nEstimated ts:'
    print np.array(bcur.ts)
    print 'Estimated Rs:'
    print np.array(bcur.Rs)
    print 'Estimated points:'
    print np.array(bcur.pts)
    print 'Estimated cost:'
    print bcur.cost()

    print '\nTrue ts:'
    print np.array(b_true.ts)
    print 'True Rs:'
    print np.array(b_true.Rs)
    print 'True points:'
    print np.array(b_true.pts)
    print 'True cost:'
    print b_true.cost()

    print '\nError in ts:'
    print abs(np.array(b_true.ts) - bcur.ts)
    print 'Error in Rs:'
    print abs(asarray(b_true.Rs) - bcur.Rs)
    print 'Error in points:'
    print sum(square(abs(array(b_true.pts) - bcur.pts)), axis=1)

    print '\nOffset in Rs:'
    for Rtrue,Rinit in zip(b_true.Rs, b_init.Rs):
        print dots(Rtrue, Rinit.T)

    #print '\nEvolution of cost function:'
    #print '  ',costs

    print '\nCost (initial -> estimated -> true)'
    print '  %f -> %f -> %f' % (b_init.cost(), bcur.cost(), b_true.cost())

    import draw_bundle
    draw_bundle.output_views(bcur, 'out/estimated.pdf')
    draw_bundle.output_views(b_init, 'out/init.pdf')
    draw_bundle.output_views(b_true, 'out/true.pdf')

############################################################################
if __name__ == '__main__':
    check_projection_jacobians()
    check_bundle_jacobians()
    
    check_schur()
    check_schur_backsub()
    check_schur_update()

    #test_optimize()
    #test_optimize2()
    test_optimize3()
