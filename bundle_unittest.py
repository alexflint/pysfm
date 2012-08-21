import unittest
import numpy as np
from numpy import *

from bundle import *
import sensor_model
import numpy_test

from copy import deepcopy

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
    NUM_POINTS         = 5
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
    msm = np.array([[ project(K, R, t, pt) for pt in pts ]
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
    b = Bundle.FromArrays(K, Rs, ts, pts, msm, msm_mask)

    # Attach robustifier
    b.sensor_model = sensor_model.CauchyModel(CAUCHY_PARAM)

    # Store this for visualization later
    b.outlier_mask = outlier_mask
    return b

################################################################################
class BundleTest(numpy_test.NumpyTestCase):
    def setUp(self):
        self.bundle = create_test_bundle()
        self.bundle.num_msm = np.sum([len(t.measurements) for t in self.bundle.tracks])

    def test_Rs(self):
        self.assertShape(self.bundle.Rs(), (4,3,3))

    def test_ts(self):
        self.assertShape(self.bundle.ts(), (4,3))

    def test_points(self):
        self.assertShape(self.bundle.points(), (5,3))

    def test_residuals(self):
        r = self.bundle.residuals()
        self.assertShape(r, (self.bundle.num_msm*2,))

    def test_Jresiduals(self):
        J,rowlabels,collabels = self.bundle.Jresiduals_extended()
        nr = self.bundle.num_msm*2
        nc = self.bundle.num_params()
        self.assertShape(J, (nr, nc))
        self.assertShape(rowlabels, (nr, 2))
        self.assertShape(collabels, (nc, 2))

    def _test_Jresidual_extended(self):
        ###
        J1 = self.bundle.Jresiduals_partial(cameras_to_include=[0],
                                           tracks_to_include=[0],
                                           cameras_to_optimize=[0],
                                           tracks_to_optimize=[])
        self.assertShape(J1, (2, 6))

        ###
        J2 = self.bundle.Jresiduals_partial(cameras_to_include=[0],
                                           tracks_to_include=[0],
                                           cameras_to_optimize=[],
                                           tracks_to_optimize=[0])
        self.assertShape(J2, (2, 3))

        ###
        J2 = self.bundle.Jresiduals_partial(cameras_to_include=[0],
                                           tracks_to_include=[0],
                                           cameras_to_optimize=[0],
                                           tracks_to_optimize=[0])
        self.assertShape(J2, (2, 9))

        ###
        J3 = self.bundle.Jresiduals_partial(cameras_to_include=[0],
                                           tracks_to_include=[0],
                                           tracks_to_optimize=[])
        self.assertShape(J3, (2, len(self.bundle.cameras)*6))

        ###
        J4 = self.bundle.Jresiduals_partial(cameras_to_include=[0],
                                           tracks_to_include=[0],
                                           cameras_to_optimize=[0,1])
        self.assertShape(J4, (2, 12 + len(self.bundle.tracks)*3))
        

    def test_pinhole_jacobians(self):
        '''Test Jacobian of the individual projection functions'''
        K0 = np.array([[ 2., 0., -1.5 ],
                       [ .1, 3., .8 ],
                       [ 0., 0., 1. ]])
        R0 = rotation_xy(1.1)
        t0 = np.array([4., 2., -1.])
        x0 = np.array([ -1., 5., 2. ])

        # Check Jacobian with respect to rotation
        self.assertJacobian(lambda m: project2(K0, R0, m, t0, x0),
                            Jproject_R(K0, R0, t0, x0),
                            np.zeros(3))

        # Check jacobian with respect to camera translation
        self.assertJacobian(lambda t: project(K0, R0, t, x0),
                            Jproject_t(K0, R0, t0, x0),
                            t0)

        # Check jacobian with respect to landmark position
        self.assertJacobian(lambda x: project(K0, R0, t0, x),
                            Jproject_x(K0, R0, t0, x0),
                            x0)


    def test_sensor_model(self):
        gm = sensor_model.GaussianModel([2., 3.])
        cm = sensor_model.CauchyModel(.4)
        
        self.assertJacobian(gm.residual_from_error,
                            gm.Jresidual_from_error,
                            [1., 2.])

        self.assertJacobian(cm.residual_from_error,
                            cm.Jresidual_from_error,
                            [1., 2.])

    def test_bundle_jacobians(self):
        '''Test Jacobian of the entire bundle'''

        self.assertJacobian(
            lambda v: deepcopy(self.bundle).perturb(v).residuals(),
            self.bundle.Jresiduals(),
            np.zeros(self.bundle.num_params()))

        # Test partial jacobians
        self.assertJacobian(
            lambda v: deepcopy(self.bundle).perturb(v).residuals_partial([0,1], [1,2,3]),
            self.bundle.Jresiduals_partial([0,1], [1,2,3]),
            np.zeros(self.bundle.num_params()))

################################################################################
if __name__ == '__main__':
    unittest.main()
