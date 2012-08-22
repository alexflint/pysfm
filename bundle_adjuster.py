from copy import copy,deepcopy

from algebra import *
import optimize
from bundle import Bundle

def select(L, mask):
    L = np.asarray(L)
    mask = np.asarray(mask)
    if mask.dtype.kind == 'b':
        assert len(mask) == len(L)
        subset = L[mask]
    else:
        assert mask.dtype == L.dtype
        assert set(mask).issubset(L), 'Mask contained some items not in L'
        subset = mask

    lookup = { v:i for i,v in enumerate(L) }
    subset_indices = [ lookup[v] for v in subset ]
    return subset,subset_indices

############################################################################
class BundleAdjuster:
    def __init__(self, bundle=None):
        self.num_steps = 0
        self.converged = False
        self.costs = []
        if bundle is not None:
            self.set_bundle(bundle)

    # Configure the bundle that this adjuster operates on.
    # camera_ids selects a subset of cameras to use measurements from
    # track_ids selects a subset of tracks to use measurements from
    # cam_mask determines which cameras get updated during each
    # gradient step. Can be a boolean array of size equal to
    # camera_ids, or a list of integers forming a subset of camera_ids.
    # track_mask is the same as cam_mask but for tracks
    def set_bundle(self,
                   bundle,
                   camera_ids=None,
                   track_ids=None,
                   camera_mask=None,
                   track_mask=None):
        self.bundle = bundle

        # Figure out the camera indices
        if camera_ids is None:
            self.camera_ids = range(len(bundle.cameras))
        else:
            self.camera_ids = list(camera_ids)
            assert self.camera_ids.dtype.kind == 'i'
            assert self.camera_ids.min() >= 0
            assert self.camera_ids.max() < len(bundle.cameras)

        # Figure out the track indices
        if track_ids is None:
            self.track_ids = range(len(bundle.tracks))
        else:
            self.track_ids = list(track_ids)
            assert self.track_ids.dtype.kind == 'i'
            assert self.track_ids.min() >= 0
            assert self.track_ids.max() < len(bundle.tracks)

        self.camera_id_set = set(self.camera_ids)

        # Figure out which subset we're optimizing
        if camera_mask is None:
            self.optim_camera_ids = self.camera_ids[1:]  # by default adjust all but the first camera
            self.optim_camera_indices = range(1,len(self.camera_ids))
        else:
            self.optim_camera_ids, self.optim_camera_indices = select(self.camera_ids, camera_mask)

        if track_mask is None:
            self.optim_track_ids = copy(self.track_ids)       # by default adjust all tracks
            self.optim_track_indices = range(1,len(self.track_ids))
        else:
            self.optim_track_ids, self.optim_track_indices = select(self.track_ids, track_mask)

        # Allocate arrays
        nc = len(self.camera_ids)
        nt = len(self.track_ids)
        self.HCCs = np.empty((nc, 6, 6))      # 6x6 diagonal blocks of top left part
        self.HCPs = np.empty((nc, nt, 6, 3))  # complete blocks of the off-diagonal part
        self.HPPs = np.empty((nt, 3, 3))      # 3x3 on-diagonal blocks of the bottom right part
        self.HPP_invs = np.empty((nt, 3, 3))  # block-wise inverse of above
        self.bCs = np.empty((nc, 6))          # top part of J.T * r     (i.e. RHS of normal eqns)
        self.bPs = np.empty((nt, 3))          # bottom part of J.T * r  (i.e. RHS of normal eqns)

    # Optimize the current bundle to convergence
    def optimize(self, param_mask, max_steps=25, init_damping=100.):
        # Begin optimizing
        damping = init_damping
        self.num_steps = 0
        self.converged = False
        self.costs = [ self.bundle.cost() ]
        while not self.converged and self.num_steps < max_steps:
            self.num_steps += 1
            cur_cost = self.bundle.cost()
            print 'Step %d: cost=%f, damping=%f' % (self.num_steps, cur_cost, damping)

            while not self.converged:
                # Compute update
                try:
                    delta = self.compute_update(damping, param_mask)
                except np.linalg.LinAlgError:
                    # Matrix was singular: increase damping
                    damping *= 10.
                    self.converged = damping > 1e+8
                    continue

                # Apply update
                bnext = deepcopy(self.bundle).perturb(delta, param_mask)
                next_cost = bnext.cost()

                # Decide whether to accept it
                if next_cost < cur_cost:
                    damping *= .1
                    self.bundle = bnext
                    self.costs.append(next_cost)
                    self.converged = abs(cur_cost - next_cost) < 1e-8
                    break
                else:
                    damping *= 10.
                    self.converged = damping > 1e+8

        if self.converged:
            print 'Converged after %d steps' % self.num_steps
        else:
            print 'Failed to converge after %d steps' % self.num_steps


    # Solve the normal equations using the schur complement.
    # Return (update-for-cameras), (update-for-points)
    def compute_update(self, damping, param_mask=None):
        nc = len(self.camera_ids)
        nt = len(self.track_ids)

        # The way we do parameter elimination here is in fact
        # mathematically equivalent to eliminating the parameters from
        # the original matrix. It is slightly inefficient though.
        if param_mask is None:
            param_mask = np.ones(6*nc + 3*nt)
        else:
            assert param_mask.dtype.kind == 'b'
            assert np.shape(param_mask) == (6*nc + 3*nt,) , \
                'shape was %s by there are %d parameters' % \
                (str(np.shape(param_mask)), self.bundle.num_params())

        # Get parameter masks for camera / point parameters
        cam_param_mask = param_mask[:nc*6 ]
        pt_param_mask =  param_mask[ nc*6:]
        assert np.all(pt_param_mask), 'Eliminating point parameters not implemented'
        
        # Compute the update
        self.prepare_schur_complement()
        self.apply_damping(damping)
        S, b = self.compute_schur_complement()
        cam_update = self.solve_camera_normal_eqns(S, b, cam_param_mask)
        point_update = self.backsubstitute(cam_update)
        return -cam_update, -point_update

    # Compute components of the Hessian that will be used in the Schur complement
    def prepare_schur_complement(self):
        bundle = self.bundle
        nc = len(self.camera_ids)
        nt = len(self.track_ids)

        # Fill with zeros
        self.HCCs.fill(0.)
        self.HPPs.fill(0.)
        self.HCPs.fill(0.)
        self.bCs.fill(0.)
        self.bPs.fill(0.)

        # Compute various components
        for j,track_id in enumerate(self.track_ids):
            track = bundle.tracks[track_id]
            for i,camera_id in enumerate(self.camera_ids):
                if track.has_measurement(camera_id):
                    r = bundle.residual(camera_id, track_id)
                    Jc, Jp = bundle.Jresidual(camera_id, track_id)
                    self.HCCs[i] += dots(Jc.T, Jc)
                    self.HPPs[j] += dots(Jp.T, Jp)
                    self.HCPs[i,j] = dots(Jc.T, Jp)
                    self.bCs[i] += dots(Jc.T, r)
                    self.bPs[j] += dots(Jp.T, r)

    # Apply levenberg-marquardt damping to the blocks along along the
    # Hessian diagonal.
    def apply_damping(self, damping):
        # Apply damping to diagonal
        for pos in range(len(self.camera_ids)):
            optimize.apply_lm_damping_inplace(self.HCCs[pos], damping)
        for pos in range(len(self.track_ids)):
            optimize.apply_lm_damping_inplace(self.HPPs[pos], damping)

    # Compute the schur complement for the normal equations. The
    # result is a matrix A and a vector b such that the gradient step
    # for the cameras is the solution to A*x=b.
    def compute_schur_complement(self):
        bundle = self.bundle

        # Invert the lower-right diagonal blocks. These are used again
        # in backsubstitute so import to invert all the right blocks.
        for pos,i in enumerate(self.track_ids):
            self.HPP_invs[pos] = np.linalg.inv(self.HPPs[pos])

        # Intialize the LHS and RHS of camera normal equations
        nc = len(self.optim_camera_ids)
        S = np.zeros((nc, nc, 6, 6))
        b = np.zeros((nc, 6))

        for pos,i in enumerate(self.optim_camera_indices):
            S[ pos, pos ] = self.HCCs[i]
            b[ pos ] = self.bCs[i]

        for k in range(len(self.track_ids)):
            # Here we iterate over all cameras, even though some
            # tracks may be measured in only a subset of cameras. The
            # Hessian blocks for those components will be zero so
            # we're guaranteed to compute the right thing, though
            # perhaps there is a more efficient way to do it.
            for ipos,i in enumerate(self.optim_camera_indices):
                b[ ipos ] -= dots(self.HCPs[i,k], self.HPP_invs[k], self.bPs[k])
                for jpos,j in enumerate(self.optim_camera_indices):
                    S[ ipos,jpos ] -= dots(self.HCPs[i,k], self.HPP_invs[k], self.HCPs[j,k].T)

        return S,b

    # Solve the normal equations for the camera system
    def solve_camera_normal_eqns(self, S, b, param_mask):
        nc = len(self.optim_camera_ids)
        assert np.shape(S) == (nc,nc,6,6)
        assert np.shape(b) == (nc,6)
        assert np.shape(param_mask) == (nc,)

        # S is a nc x nc array where each element is a 6x6 matrix. So
        # it has 4 dimensions. We want to "flatten" it to a 6*nc x
        # 6*nc matrix.
        AC = S.transpose((0,2,1,3)).reshape((nc*6,nc*6))
        bC = b.flatten()

        # Apply the parameter mask
        AC_reduced = AC[param_mask].T[param_mask].T
        bC_reduced = bC[param_mask]

        # Solve the system
        dC_reduced = np.linalg.solve(AC_reduced, bC_reduced)
        
        # Reverse the parameter mask
        dC = np.zeros(nc*6)
        dC[param_mask] = dC_reduced

        # Split into per-camera updates
        return dC.reshape(nc,6)
        

    # Backsubstitute a solution for the camera update into the normal
    # equations to get a list of updates for each 3D point.
    def backsubstitute(self, dC):
        bundle = self.bundle

        # Compute updates
        nt = len(self.optim_track_ids)
        dP = np.zeros((nt, 3))
        for ipos,i in enumerate(self.optim_tracks_indices):
            # Here we again sum over all cameras, even though some
            # cameras may not have observed this track. But the
            # Hessian blocks for those will be zero so we're still in
            # good shape.
            dP[ipos] = self.bPs[i]
            for j,jpos in enumerate(self.optim_camera_indices):
                dP[ipos] -= dots(self.HCPs[j,i].T, dC[jpos])
            dP[ipos] = dots(self.HPP_invs[i], dP[ipos])

        return dP
