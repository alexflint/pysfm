from algebra import *
import lie
import sensor_model
import optimize

############################################################################
# Jacobian of projection operation, evaluated at X
def Jpr(x):
    x = np.asarray(x)
    return np.array([[ 1./x[2],    0,         -x[0] / (x[2]*x[2])],
                     [ 0,          1./x[2],   -x[1] / (x[2]*x[2])]])

# Pinhole projection
def project(K, R, t, x):
    K = np.asarray(K)
    R = np.asarray(R)
    t = np.asarray(t)
    x = np.asarray(x)
    return pr(dots(K, R, x) + dots(K, t))

# Pinhole project with rotations represented in the tangent space at R0
def project2(K, R0, m, t, x):
    R = dots(R0, lie.SO3.exp(m))
    return project(K, R, t, x)

# Jacobian of project w.r.t. landmark
def Jproject_x(K, R, t, x):
    return dots(Jpr(dots(K, R, x) + dots(K, t)), K, R)

# Jacobian of project w.r.t. translation
def Jproject_t(K, R, t, x):
    return dots(Jpr(dots(K, R, x) + dots(K, t)), K)

# Jacobian of project w.r.t. rotation
def Jproject_R(K, R, t, x):
    return np.dot(Jpr(dots(K, R, x) + dots(K, t)),
                  dots(K, R, lie.SO3.J_expm_x(x)))

# Jacobian of projection w.r.t. camera params
def Jproject_cam(K, R, t, x):
    return np.hstack((Jproject_R(K, R, t, x),
                      Jproject_t(K, R, t, x)))

############################################################################
# Represents a pinhole camera with a rotation and translation
class Camera(object):
    def __init__(self, R=None, t=None, idx=None):
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros(3)
        assert np.shape(R) == (3,3)
        assert np.shape(t) == (3,)
        self.idx = idx
        self.R = np.asarray(R)
        self.t = np.asarray(t)
        assert self.R
        
    def projection_matrix(self):
        return np.hstack((self.R, self.t[:,np.newaxis]))

    def clone(self):
        return Camera(self.R, self.t, self.idx)

    def update
    
############################################################################
# Represents a set of measurements associated with a single 3D point
class Track(object):
    def __init__(self, camera_ids=[], measurements=[], reconstruction=None):
        assert isinstance(camera_ids, list)
        assert isinstance(measuremets_ids, list)
        assert len(camera_ids) == len(measurements)
        if reconstruction is None:
            reconstruction = np.zeros(3)
        self.measurements = dict(zip(camera_ids, measurements))
        self.reconstruction = reconstruction

    # Add a measurement to this track for the camera with the given ID
    def add_measurement(self, camera_id, measurement):
        assert np.shape(measurement) == (2,)
        assert type(camera_id) is int
        self.measurements[ camera_id ] = measurement

    def has_measurement(self, camera_id):
        return camera_id in self.measurements

    def camera_ids(self):
        return self.measurements.viewkeys()

    def clone(self):
        clone = Track(self.measurements.viewvalues(),
                      self.measurements.viewvalues(),
                      self.reconstruction)
        return clone

############################################################################
# Represents a set of cameras and points
class Bundle(object):
    NumCamParams = 6
    NumPointParams = 3

    def __init__(self):
        self.cameras = []
        self.tracks = []
        self.K = np.eye(3)


        self.ncameras = ncameras
        self.npts = npts
        self.Rs = [ np.eye(3) ] * ncameras
        self.ts = [ np.zeros(3) ] * ncameras
        self.pts = [ np.zeros(3) ] * npts
        self.msm = np.zeros((ncameras, npts, 2))
        self.msm_mask = np.ones((ncameras, npts), bool)
        self.sensor_model = sensor_model.GaussianModel(1.)

    # Check sizes etc
    def check_consistency(self):
        assert np.shape(self.K) == (3, 3), \
            'shape was '+str(np.shape(self.K))

    # Get a list of rotation matrices for all cameras
    def Rs():
        return np.array([ cam.R for cam in cameras ])

    # Get a list of translation vectors for all cameras
    def ts():
        return np.array([ cam.t for cam in cameras ])

    # Get the measurement of the j-th track in the i-th camera
    def measurement(self, i, j):
        return self.tracks[j].get_measurement(i)

    # Get (camer_id, track_id) pairs for all measurements
    def measurement_ids(self, i, j):
        return ( (i,j) for j in range(len(tracks)) for i in trakcs[j].camera_ids() )

    # Get the number of free parameters in the entire system
    def num_params():
        return len(self.cameras) * Bundle.NumCamParams + len(self.tracks) * Bundle.NumPointParams

    # Get the prediction for the j-th track in the i-th camera
    def predict(self, i, j):
        assert self.msm_mask[i,j]
        return project(self.K, self.cameras[i].R, self.cameras[i].t, self.tracks[j].reconstruction)

    # Get the element-wise difference between a measurement and its prediction
    def reproj_error(self, i, j):
        return self.predict(i,j) - self.measurement(i,j)

    # Get the component of the residual for the measurement of point j in camera i
    def residual(self, i, j):
        return self.sensor_model.residual_from_error(self.reproj_error(i, j))

    # Get the jacobian of the residual for camera i, track j
    def Jresidual(self, i, j):
        R = self.cameras[i].R
        t = self.cameras[i].t
        x = self.tracks[j].reconstruction
        Jcost_r = self.sensor_model.Jresidual_from_error(self.reproj_error(i, j))
        Jr_cam = dots(Jcost_r, Jproject_cam(self.K, R, t, x))
        Jr_x = dots(Jcost_r, Jproject_x(self.K, R, t, x))
        return Jr_cam, Jr_x

    # Get an array containing predictions for each (camera, point) pair
    def predictions(self):
        return np.array([ self.predict(i,j)
                          for (i,j) in self.measurement_ids() ])

    # Get an array containing predictions for each (camera, point) pair
    def reproj_errors(self):
        return np.array([ self.reproj_error(i,j) 
                          for (i,j) in self.measurement_ids() ])

    # Get the complete residual vector. It is a vector of length NUM_MEASUREMENTS*2
    # where NUM_MEASUREMENTS is the sum of the track lengths
    def residuals(self):
        return np.array([ self.residual(i,j)
                          for (i,j) in self.measurement_ids() ])

    # Get the Jacobian of the complete residual vector
    def Jresiduals(self):
        msm_ids = list(self.measurement_ids)
        J = np.zeros((len(msm_ids)*2, self.num_params()))
        for msm_idx,(i,j) in enumerate(zip(*np.nonzero(self.msm_mask))):
            # Compute jacobians
            Jr_cam, Jr_x = self.Jresidual(i,j)

            # Slot into the main array
            row = msm_idx*2
            camcol = i * Bundle.NumCamParams
            ptcol = self.ncameras * Bundle.NumCamParams + j * Bundle.NumPointParams
            J[ row:row+2, camcol:camcol+Bundle.NumCamParams  ] = Jr_cam
            J[ row:row+2,  ptcol:ptcol+Bundle.NumPointParams ] = Jr_x

        return J

    # Get the total cost of the system
    def cost(self):
        return np.sum(np.square(self.residuals()))

    # Clone this object
    def clone(self):
        clone = Bundle()
        clone.K = self.K.copy()
        clone.tracks = [ t.clone() for t in self.tracks ]
        clone.cameras = [ cam.clone for cam in self.cameras ]
        clone.sensor_model = self.sensor_model
        return clone

    # Apply a linear update to all cameras and points
    def apply_update(self, delta, param_mask=None):
        delta = np.asarray(delta)

        if param_mask is not None:
            assert len(param_mask) == self.nparams, \
                'shape was '+str(np.shape(param_mask))

            param_mask = np.asarray(param_mask)
            reduced_delta = delta.copy()
            delta = np.zeros(self.nparams)
            delta[param_mask] = reduced_delta

        assert delta.shape == (self.nparams,), 'shape was '+str(np.shape(delta))

        for i,cam in enumerate(self.cameras):
            cam.update( delta[ i*Bundle.NumCamParams : (i+1)*Bundle.NumCamParams ] )

        offs = self.ncameras*Bundle.NumCamParams
        for i,track in enumerate(self.tracks):
            track.update( delta[ offs+i*Bundle.NumPointParams : offs+(i+1)*Bundle.NumPointParams ] )

    # Clone this object and apply a linear update to the clone
    def copy_and_update(self, delta, param_mask=None):
        clone = self.clone()
        clone.apply_update(delta, param_mask)
        return clone



############################################################################
class BundleAdjuster:
    def __init__(self, bundle):
        self.b = bundle
        nc = bundle.ncameras
        npts = bundle.npts
        self.HCCs = np.empty((nc, 6, 6))        # 6x6 diagonal blocks of top left part
        self.HCPs = np.empty((nc, npts, 6, 3))  # complete blocks of the off-diagonal part
        self.HPPs = np.empty((npts, 3, 3))      # 3x3 on-diagonal blocks of the bottom right part
        self.HPP_invs = np.empty((npts, 3, 3))  # block-wise inverse of above
        self.bCs = np.empty((nc, 6))            # top part of J.T * r     (i.e. RHS of normal eqns)
        self.bPs = np.empty((npts, 3))          # bottom part of J.T * r  (i.e. RHS of normal eqns)

    # Solve the normal equations using the schur complement.
    # Return (update-for-cameras), (update-for-points)
    def compute_update(self, bundle, damping, param_mask=None):
        self.set_bundle(bundle)

        # The way we do parameter elimination here is in fact
        # mathematically equivalent to eliminating the parameters from
        # the original matrix. It is slightly inefficient though.
        if param_mask is None:
            param_mask = np.ones(self.b.nparams).astype(bool)
        else:
            assert param_mask.dtype == bool
            assert np.shape(param_mask) == (self.b.nparams,) , \
                'shape was %s by there are %d parameters' % \
                (str(np.shape(param_mask)), self.b.nparams)

        # Get parameter masks for camera / point parameters
        cam_param_mask = param_mask[:self.b.ncameras*6 ]
        pt_param_mask =  param_mask[ self.b.ncameras*6:]
        assert np.all(pt_param_mask), 'Eliminating point parameters not implemented'
        
        # Compute schur complement
        self.prepare_schur_complement(damping)
        AC, bC = self.compute_schur_complement()

        # Eliminate some parameters
        AC = AC[cam_param_mask].T[cam_param_mask].T
        bC = bC[cam_param_mask]

        # Solve normal equations and backsubstitute
        dC = np.zeros(bundle.ncameras*6)
        dC[cam_param_mask] = np.linalg.solve(AC, bC)
        dP = self.backsubstitute(dC)
        return -np.concatenate((dC[cam_param_mask], dP))

    # Configure the bundle that this adjuster operates on
    def set_bundle(self, bundle):
        assert bundle.ncameras == self.b.ncameras, 'Bundle must not change number of parameters'
        assert bundle.npts == self.b.npts, 'Bundle must not change number of parameters'
        self.b = bundle

    # Compute components of the Hessian that will be used in the Schur complement
    def prepare_schur_complement(self, damping=0.):
        nc = self.b.ncameras
        npts = self.b.npts

        # Fill with zeros in preparation for summing
        self.HCCs.fill(0.)
        self.HPPs.fill(0.)
        self.HCPs.fill(0.)
        self.bCs.fill(0.)
        self.bPs.fill(0.)

        # Compute various components
        for i,j in zip(*np.nonzero(self.b.msm_mask)):
            err_ij = self.b.reproj_error(i, j)
            r = self.b.sensor_model.residual_from_error(err_ij)
            Jcost_r = self.b.sensor_model.Jresidual_from_error(err_ij)
            Jr_cam = Jproject_cam(self.b.K, self.b.Rs[i], self.b.ts[i], self.b.pts[j])
            Jr_pt = Jproject_x(self.b.K, self.b.Rs[i], self.b.ts[i], self.b.pts[j])
            Jc = dots(Jcost_r, Jr_cam)
            Jp = dots(Jcost_r, Jr_pt)
            self.HCCs[i] += dots(Jc.T, Jc)
            self.HPPs[j] += dots(Jp.T, Jp)
            self.HCPs[i,j] = dots(Jc.T, Jp)
            self.bCs[i] += dots(Jc.T, r)
            self.bPs[j] += dots(Jp.T, r)

        # Apply damping to diagonal
        if damping > 0.:
            for i in range(nc):
                optimize.apply_lm_damping_inplace(self.HCCs[i], damping)
            for i in range(npts):
                optimize.apply_lm_damping_inplace(self.HPPs[i], damping)

        # Invert the lower right sub-matrix (which is block-diagonal)
        for j in range(npts):
            self.HPP_invs[j] = np.linalg.inv(self.HPPs[j])

    # Compute the schur complement for the normal equations
    def compute_schur_complement(self, damping=0.):
        nc = self.b.ncameras
        npts = self.b.npts

        S = np.zeros((nc*6, nc*6))
        b = np.zeros(nc*6)

        for i in range(nc):
            S[ i*6:(i+1)*6, i*6:(i+1)*6 ] = self.HCCs[i]
            b[ i*6:(i+1)*6 ] = self.bCs[i]

        for k in range(npts):
            # Find cameras that measured this point
            msm_cameras = np.nonzero(self.b.msm_mask[:,k])[0]
            for i in msm_cameras:
                b[ i*6:(i+1)*6 ] -= dots(self.HCPs[i,k], self.HPP_invs[k], self.bPs[k])
                for j in msm_cameras:
                    S[ i*6:(i+1)*6, j*6:(j+1)*6 ] -= \
                        dots(self.HCPs[i,k], self.HPP_invs[k], self.HCPs[j,k].T)

        return S,b

    # Compute the update for the points from the update for the cameras
    def backsubstitute(self, dC):
        dP = np.zeros(self.b.npts*3)
        for i in range(self.b.npts):
            # Find cameras that measured this point
            msm_cameras = np.nonzero(self.b.msm_mask[:,i])[0]

            dPi = self.bPs[i]
            for j in msm_cameras:
                dPi -= dots(self.HCPs[j,i].T, dC[j*6 : (j+1)*6])
            dPi = dots(self.HPP_invs[i], dPi)
            dP[ i*3 : (i+1)*3 ] = dPi

        return dP
