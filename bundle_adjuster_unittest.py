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
        self.bundle.tracks = [ self.bundle.tracks[0] ]
        self.bundle.cameras = [ self.bundle.cameras[0] ]
        for track in self.bundle.tracks:
            track.measurements = { 0: track.measurements[0] }

    ############################################################################
    def test_schur_compliment(self):
        bundle = self.bundle
        damping = 0
    
        r = bundle.residuals()
        J = bundle.Jresiduals();
        JTJ = dots(J.T, J)
        JTr = dots(J.T, r)

        optimize.apply_lm_damping_inplace(JTJ, damping)
        Aslow, bslow = schur.get_schur_complement(JTJ, JTr, 6*len(bundle.cameras))
    
        ba = BundleAdjuster(bundle)
        ba.prepare_schur_complement()
        ba.apply_damping(damping)
        A,b = ba.compute_schur_complement()

        self.assertArrayEqual(A, Aslow)
        self.assertArrayEqual(b, bslow)
    
    ############################################################################
    def test_schur_backsub(self):
        bundle = self.bundle
        damping = 2.

        # Compute terms explicitly, slowly
        r = bundle.residuals()
        J = bundle.Jresiduals();
        JTJ = dots(J.T, J)
        JTr = dots(J.T, r)

        optimize.apply_lm_damping_inplace(JTJ, damping)
        delta_slow = -np.linalg.solve(JTJ, JTr);

        # Compute terms again, using wrapper
        ba = BundleAdjuster(bundle)
        delta = ba.compute_update(damping)
        self.assertArrayEqual(delta, delta_slow)


if __name__ == '__main__':
    unittest.main()
