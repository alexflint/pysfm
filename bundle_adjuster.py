from copy import copy,deepcopy
from numpy import *
from numpy.linalg import *
import numpy as np

from algebra import *
import optimize
from bundle import Bundle

def select(L, mask):
    L = asarray(L)
    mask = asarray(mask)
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
        bundle.check_consistency()  # will throw AssertionError if anything is awry
        self.bundle = bundle

        # Figure out the camera indices
        if camera_ids is None:
            self.camera_ids = range(len(bundle.cameras))
        else:
            self.camera_ids = list(camera_ids)
            assert type(self.camera_ids[0]) is int
            assert min(self.camera_ids) >= 0
            assert max(self.camera_ids) < len(bundle.cameras)

        # Figure out the track indices
        if track_ids is None:
            self.track_ids = range(len(bundle.tracks))
        else:
            self.track_ids = list(track_ids)
            assert type(self.track_ids[0]) is int
            assert min(self.track_ids) >= 0
            assert max(self.track_ids) < len(bundle.tracks)

        self.camera_id_set = set(self.camera_ids)

        # Figure out which subset we're optimizing
        if camera_mask is None:
            assert len(self.camera_ids) > 1, 'Cannot optimize just one camera'
            # by default adjust all but the first camera
            self.optim_camera_ids = self.camera_ids[1:]  
            self.optim_camera_indices = range(1,len(self.camera_ids))
        else:
            self.optim_camera_ids, self.optim_camera_indices = select(self.camera_ids, camera_mask)

        if track_mask is None:
            # by default adjust all tracks
            self.optim_track_ids = copy(self.track_ids)
            self.optim_track_indices = range(len(self.track_ids))
        else:
            self.optim_track_ids, self.optim_track_indices = select(self.track_ids, track_mask)

        # Consistency check
        assert len(self.optim_track_ids) == len(self.optim_track_indices)
        assert len(self.optim_camera_ids) == len(self.optim_camera_indices)

        # Allocate arrays
        nc = len(self.camera_ids)
        nt = len(self.track_ids)
        self.HCCs     = empty((nc, 6, 6))      # 6x6 diagonal blocks of top left part
        self.HCPs     = empty((nc, nt, 6, 3))  # complete blocks of the off-diagonal part
        self.HPPs     = empty((nt, 3, 3))      # 3x3 on-diagonal blocks of the bottom right part
        self.HPP_invs = empty((nt, 3, 3))      # block-wise inverse of above
        self.bCs      = empty((nc, 6))         # top part of J.T * r     (i.e. RHS of normal eqns)
        self.bPs      = empty((nt, 3))         # bottom part of J.T * r  (i.e. RHS of normal eqns)

        # Print info
        print 'Configured a bundle adjuster for %d cameras, %d tracks' % (nc, nt)

    # Optimize the current bundle to convergence
    def optimize(self, param_mask=None, max_steps=25, init_damping=100.):
        # Begin optimizing
        damping = init_damping
        self.num_steps = 0
        self.converged = False
        self.costs = [ self.compute_cost(self.bundle) ]
        while not self.converged and self.num_steps < max_steps:
            self.num_steps += 1
            cur_cost = self.compute_cost(self.bundle)
            print 'Step %d: cost=%f, damping=%f' % (self.num_steps, cur_cost, damping)

            while not self.converged:
                # Compute update
                try:
                    motion_update,structure_update = self.compute_update(damping, param_mask)
                except LinAlgError:
                    # Matrix was singular: increase damping
                    damping *= 10.
                    self.converged = damping > 1e+8
                    continue

                # Apply update
                bnext = deepcopy(self.bundle)
                self.update_motion(motion_update, bnext)
                self.update_structure(structure_update, bnext)
                next_cost = self.compute_cost(bnext)

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

    # Compute current cost.
    def compute_cost(self, bundle):
        cost = 0.
        for j in self.optim_track_ids:
            for i in self.optim_camera_ids:
                if bundle.tracks[j].has_measurement(i):
                    cost += sum(square(bundle.residual(i,j)))
        return cost


    # Solve the normal equations using the schur complement.
    # Return (update-for-cameras), (update-for-points)
    def compute_update(self, damping, param_mask=None):
        nc = len(self.optim_camera_ids)
        nt = len(self.optim_track_ids)

        # The way we do parameter elimination here is in fact
        # mathematically equivalent to eliminating the parameters from
        # the original matrix. It is slightly inefficient though.
        if param_mask is None:
            param_mask = ones(6*nc + 3*nt, bool)
        else:
            assert param_mask.dtype.kind == 'b'
            assert shape(param_mask) == (6*nc + 3*nt,) , \
                'shape was %s by there are %d parameters' % \
                (str(shape(param_mask)), self.bundle.num_params())

        # Get parameter masks for camera / point parameters
        cam_param_mask = param_mask[:nc*6 ]
        pt_param_mask =  param_mask[ nc*6:]
        assert all(pt_param_mask), 'Eliminating point parameters not implemented'
        
        # Compute the update
        self.prepare_schur_complement()
        self.apply_damping(damping)
        S, b = self.compute_schur_complement()
        motion_update = self.solve_motion_normal_eqns(S, b, cam_param_mask)
        structure_update = self.backsubstitute(motion_update)
        return -motion_update, -structure_update

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
                    self.HCCs[i]    += dots(Jc.T, Jc)
                    self.HPPs[j]    += dots(Jp.T, Jp)
                    self.HCPs[i,j]   = dots(Jc.T, Jp)
                    self.bCs[i]     += dots(Jc.T, r)
                    self.bPs[j]     += dots(Jp.T, r)

    # Apply levenberg-marquardt damping to the blocks along along the
    # Hessian diagonal.
    def apply_damping(self, damping):
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
            self.HPP_invs[pos] = inv(self.HPPs[pos])

        # Intialize the LHS and RHS of camera normal equations
        nc = len(self.optim_camera_ids)
        S = zeros((nc, nc, 6, 6))
        b = zeros((nc, 6))

        for pos,i in enumerate(self.optim_camera_indices):
            S[ pos, pos ] = self.HCCs[i]
            b[ pos ]      = self.bCs[i]

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
    def solve_motion_normal_eqns(self, S, b, param_mask):
        nc = len(self.optim_camera_ids)
        assert shape(S) == (nc,nc,6,6)
        assert shape(b) == (nc,6)
        assert shape(param_mask) == (nc*6,), 'shape was '+str(shape(param_mask))

        # S is a nc x nc array where each element is a 6x6 matrix. So
        # it has 4 dimensions. We want to "flatten" it to a 6*nc x
        # 6*nc matrix.
        AC = S.transpose((0,2,1,3)).reshape((nc*6,nc*6))
        bC = b.flatten()

        # Apply the parameter mask
        AC_reduced = AC[param_mask].T[param_mask].T
        bC_reduced = bC[param_mask]

        # Solve the system
        dC_reduced = solve(AC_reduced, bC_reduced)
        
        # Reverse the parameter mask
        dC = zeros(nc*6)
        dC[param_mask] = dC_reduced

        # Split into per-camera updates
        return dC.reshape(nc,6)

    # Backsubstitute a solution for the camera update into the normal
    # equations to get a list of updates for each 3D point.
    def backsubstitute(self, dC):
        nt = len(self.optim_track_ids)
        dP = zeros((nt, 3))

        for ipos,i in enumerate(self.optim_track_indices):
            # Here we again sum over all cameras, even though some
            # cameras may not have observed this track. But the
            # Hessian blocks for those will be zero so we're in good
            # shape regardless.
            dP[ipos] = self.bPs[i]
            for jpos,j in enumerate(self.optim_camera_indices):
                dP[ipos] -= dots(self.HCPs[j,i].T, dC[jpos])
            dP[ipos] = dots(self.HPP_invs[i], dP[ipos])

        return dP

    # Update camera parameters given the solution to the normal equations
    def update_motion(self, delta, bundle):
        assert shape(delta) == (len(self.optim_camera_ids), 6)
        for pos,idx in enumerate(self.optim_camera_ids):
            bundle.cameras[idx].perturb(delta[pos])

    # Update 3D point parameters given the solution to the normal equations
    def update_structure(self, delta, bundle):
        assert shape(delta) == (len(self.optim_track_ids), 3)
        for pos,idx in enumerate(self.optim_track_ids):
            bundle.tracks[idx].perturb(delta[pos])

