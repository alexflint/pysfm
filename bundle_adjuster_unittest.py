import unittest
import numpy as np
from numpy import *

import schur
import numpy_test
import bundle_unittest
import optimize
from bundle_adjuster import *

class BundleAdjusterTest(numpy_test.NumpyTestCase):
    def setUp(self):
        self.bundle = bundle_unittest.create_test_bundle()

    ############################################################################
    def test_schur_compliment(self):
        damping = 0.
        bundle = self.bundle
    
        r = bundle.residuals()   # do not adjust the first camera
        J = bundle.Jresiduals()[:,6:]  # do not adjust the first camera
        JTJ = dots(J.T, J)
        JTr = dots(J.T, r)
    
        n = 6*(len(bundle.cameras)-1)
        #print '\n'.join([''.join(row) for row in np.array([' ','x']).take(JTJ>1e-5)])
        #print np.round(JTr,2)
        #print n
    
        optimize.apply_lm_damping_inplace(JTJ, damping)
        Aslow, bslow = schur.get_schur_complement(JTJ, JTr, n)
    
        ba = BundleAdjuster(bundle)
        ba.prepare_schur_complement()
        ba.apply_damping(damping)
        A,b = ba.compute_schur_complement()

        nc = len(ba.optim_camera_ids)
        #print 'nc=%d'%nc
        A = A.transpose((0,2,1,3)).reshape((6*nc, 6*nc))
        b = b.flatten()

        self.assertArrayEqual(A, Aslow)
        self.assertArrayEqual(b, bslow)
    
    ############################################################################
    def test_schur_backsub(self):
        bundle = self.bundle
        damping = 2.

        # Compute terms explicitly for reference
        r = bundle.residuals()
        J = bundle.Jresiduals()[:,6:]   # do not optimize the first camera
        JTJ = dots(J.T, J)
        JTr = dots(J.T, r)

        optimize.apply_lm_damping_inplace(JTJ, damping)
        delta_slow = -np.linalg.solve(JTJ, JTr);

        # Compute terms again, using wrapper
        ba = BundleAdjuster(bundle)
        structure_update,motion_update = ba.compute_update(damping)
    
        delta = np.concatenate((np.array(structure_update).flatten(),
                                np.array(motion_update).flatten()))
    
        self.assertArrayEqual(delta, delta_slow)

    ############################################################################
    def test_subset_schur(self):
        bundle = self.bundle
        damping = 2.
    
        camera_ids = [ 3,1 ]
        track_ids =  [ 0,1,2 ]

        cam_mask   = [ False, True ]
        track_mask = [ False, True, False ]

        cameras_to_optimize = list(np.array(camera_ids)[np.array(cam_mask)])
        tracks_to_optimize = list(np.array(track_ids)[np.array(track_mask)])

        # Compute normal equations
        ba = BundleAdjuster()
        ba.set_bundle(bundle, camera_ids, track_ids, cam_mask, track_mask)
        ba.prepare_schur_complement()
        ba.apply_damping(damping)
        A,b = ba.compute_schur_complement()

        # Flatten the arrays
        nc = np.sum(cam_mask)
        self.assertShape(A, (nc,nc,6,6))
        self.assertShape(b, (nc,6))
        Aflat = A.transpose((0,2,1,3)).reshape((nc*6, nc*6))
        bflat = b.flatten()

        # Compute terms explicitly for reference
    
        # Note that for equivalence with BundleAdjuster, we must
        # compute residuals with *all* selected cameras and tracks
        # (i.e. ignoring cam_mask and track_mask), then compute the
        # Schur compliment, then eliminate rows and columns from the
        # normal equations *afterwards*
        r = bundle.residuals_partial(camera_ids, track_ids)
        J = bundle.Jresiduals_partial(camera_ids, track_ids)
        JTJ = dots(J.T, J)
        JTr = dots(J.T, r)
        optimize.apply_lm_damping_inplace(JTJ, damping)
        Aslow, bslow = schur.get_schur_complement(JTJ, JTr, len(camera_ids)*6)

        # Apply masking now
        cam_param_mask = np.array([ cam_mask[i/6] for i in range(len(camera_ids)*6) ])
        track_param_mask = np.array([ track_mask[i/3] for i in range(len(track_ids)*3) ])
        Aslow = Aslow[cam_param_mask].T[cam_param_mask].T
        bslow = bslow[cam_param_mask]

        # Check the results
        self.assertArrayEqual(Aflat, Aslow)
        self.assertArrayEqual(bflat, bslow)


if __name__ == '__main__':
    unittest.main()
