from algebra import *
import optimize
from bundle import Bundle

############################################################################
class BundleAdjuster:
    def __init__(self, bundle):
        self.bundle = bundle
        nc = len(bundle.cameras)
        nt = len(bundle.tracks)
        self.HCCs = np.empty((nc, 6, 6))        # 6x6 diagonal blocks of top left part
        self.HCPs = np.empty((nc, nt, 6, 3))  # complete blocks of the off-diagonal part
        self.HPPs = np.empty((nt, 3, 3))      # 3x3 on-diagonal blocks of the bottom right part
        self.HPP_invs = np.empty((nt, 3, 3))  # block-wise inverse of above
        self.bCs = np.empty((nc, 6))            # top part of J.T * r     (i.e. RHS of normal eqns)
        self.bPs = np.empty((nt, 3))          # bottom part of J.T * r  (i.e. RHS of normal eqns)

    # Solve the normal equations using the schur complement.
    # Return (update-for-cameras), (update-for-points)
    def compute_update(self, bundle, damping, param_mask=None):
        self.set_bundle(bundle)
        nc = len(bundle.cameras)
        nt = len(bundle.tracks)

        # The way we do parameter elimination here is in fact
        # mathematically equivalent to eliminating the parameters from
        # the original matrix. It is slightly inefficient though.
        if param_mask is None:
            param_mask = np.ones(self.bundle.num_params()).astype(bool)
        else:
            assert param_mask.dtype == bool
            assert np.shape(param_mask) == (self.b.nparams,) , \
                'shape was %s by there are %d parameters' % \
                (str(np.shape(param_mask)), self.b.nparams)

        # Get parameter masks for camera / point parameters
        cam_param_mask = param_mask[:nc*6 ]
        pt_param_mask =  param_mask[ nc*6:]
        assert np.all(pt_param_mask), 'Eliminating point parameters not implemented'
        
        # Compute schur complement
        self.prepare_schur_complement()
        AC, bC = self.compute_schur_complement(damping)

        # Eliminate some parameters
        AC = AC[cam_param_mask].T[cam_param_mask].T
        bC = bC[cam_param_mask]

        # Solve normal equations and backsubstitute
        dC = np.zeros(nc * Bundle.NumCamParams)
        dC[cam_param_mask] = np.linalg.solve(AC, bC)
        dP = self.backsubstitute(dC)
        return -np.concatenate((dC[cam_param_mask], dP))

    # Configure the bundle that this adjuster operates on
    def set_bundle(self, bundle):
        assert(len(bundle.cameras) == self.HCCs.shape[0])
        assert(len(bundle.tracks) == self.HPPs.shape[0])
        self.bundle = bundle

    # Compute components of the Hessian that will be used in the Schur complement
    def prepare_schur_complement(self):
        bundle = self.bundle
        nc = len(bundle.cameras)
        nt = len(bundle.tracks)

        # Fill with zeros
        self.HCCs.fill(0.)
        self.HPPs.fill(0.)
        self.HCPs.fill(0.)
        self.bCs.fill(0.)
        self.bPs.fill(0.)

        # Compute various components
        for i,j in bundle.measurement_ids():
            err = bundle.reproj_error(i, j)
            r = bundle.sensor_model.residual_from_error(err)
            Jc, Jp = bundle.Jresidual(i,j)
            self.HCCs[i] += dots(Jc.T, Jc)
            self.HPPs[j] += dots(Jp.T, Jp)
            self.HCPs[i,j] = dots(Jc.T, Jp)
            self.bCs[i] += dots(Jc.T, r)
            self.bPs[j] += dots(Jp.T, r)

    # Compute the schur complement for the normal equations
    def compute_schur_complement(self, damping=0.):
        bundle = self.bundle
        nc = len(bundle.cameras)
        nt = len(bundle.tracks)

        # Apply damping to diagonal
        if damping > 0.:
            for i in range(nc):
                optimize.apply_lm_damping_inplace(self.HCCs[i], damping)
            for i in range(nt):
                optimize.apply_lm_damping_inplace(self.HPPs[i], damping)

        # Invert the lower right sub-matrix (which is block-diagonal)
        for j in range(nt):
            self.HPP_invs[j] = np.linalg.inv(self.HPPs[j])

        # Intialize the LHS and RHS of normal equations for camera
        S = np.zeros((nc*6, nc*6))
        b = np.zeros(nc*6)

        for i in range(nc):
            S[ i*6:(i+1)*6, i*6:(i+1)*6 ] = self.HCCs[i]
            b[ i*6:(i+1)*6 ] = self.bCs[i]

        for k in range(nt):
            # Find cameras that measured this point
            #for i in bundle.tracks[k].camera_ids():
            for i in range(nc):
                b[ i*6:(i+1)*6 ] -= dots(self.HCPs[i,k], self.HPP_invs[k], self.bPs[k])
                #for j in bundle.tracks[k].camera_ids():
                for j in range(nc):
                    S[ i*6:(i+1)*6, j*6:(j+1)*6 ] -= \
                        dots(self.HCPs[i,k], self.HPP_invs[k], self.HCPs[j,k].T)

        return S,b

    # Compute the update for the points from the update for the cameras
    def backsubstitute(self, dC):
        bundle = self.bundle
        dP = np.zeros(len(bundle.tracks) * 3)
        for i in range(len(bundle.tracks)):
            # Note that the terms for cameras that do not observe this point will be zero!
            dPi = self.bPs[i]
            for j in range(len(bundle.cameras)):
                dPi -= dots(self.HCPs[j,i].T, dC[j*6 : (j+1)*6])
            dPi = dots(self.HPP_invs[i], dPi)
            dP[ i*3 : (i+1)*3 ] = dPi

        return dP
