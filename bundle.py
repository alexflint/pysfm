from algebra import *
import lie
import sensor_model
import triangulate

############################################################################
# Jacobian of homogeneous projection operation -- i.e. algebra.pr(...)
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
        
    # Create a 3x4 projection matrix from the rotation and translation
    def projection_matrix(self):
        return np.hstack((self.R, self.t[:,np.newaxis]))

    # Apply a linear update
    def perturb(self, delta):
        assert np.shape(delta) == (6,)
        self.R = np.dot(self.R, lie.SO3.exp(delta[:3]))
        self.t += delta[3:]
        return self

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'Camera(%s)' % \
            str(self.projection_matrix()).replace('\n','\n       ')
    
############################################################################
# Represents a set of measurements associated with a single 3D point
class Track(object):
    def __init__(self, camera_ids=[], measurements=[], reconstruction=None):
        assert isinstance(camera_ids, list)
        assert len(camera_ids) == len(measurements)
        self.measurements = dict(zip(camera_ids, measurements))
        self.reconstruction = reconstruction  # will be None if no reconstruction provided

    # Add a measurement to this track for the camera with the given ID
    def add_measurement(self, camera_id, measurement):
        assert np.shape(measurement) == (2,)
        assert type(camera_id) is int
        self.measurements[ camera_id ] = measurement

    def has_measurement(self, camera_id):
        return camera_id in self.measurements

    def get_measurement(self, camera_id):
        return self.measurements[camera_id]

    def camera_ids(self):
        return self.measurements.viewkeys()

    # Get all camera ids that are in this track and also in camera_ids
    def intersect_camera_ids(self, camera_ids):
        return self.measurements.viewkeys() & camera_ids  # set intersections

    # Apply a linear update to this object
    def perturb(self, delta):
        assert np.shape(delta) == (3,)
        self.reconstruction += delta
        return self

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'Track(%s)' % (\
            '\n      '.join([ '%-2d ->  [%10f, %10f]' % (i,m[0],m[1]) \
                                  for i,m in self.measurements.iteritems() ]))

############################################################################
# Represents a set of cameras and points
class Bundle(object):
    NumCamParams = 6
    NumPointParams = 3

    def __init__(self, ncameras=0, ntracks=0):
        self.cameras = []
        self.tracks = []
        self.K = np.eye(3)
        self.sensor_model = sensor_model.GaussianModel(1.)

        for i in range(ncameras):
            self.add_camera()

        for i in range(ntracks):
            self.add_track()

    # Check sizes etc
    def check_consistency(self):
        assert self.sensor_model is not None

        assert np.shape(self.K) == (3, 3), 'shape was '+str(np.shape(self.K))

        for i,track in enumerate(self.tracks):
            assert track.reconstruction is not None, \
                'Reconstruction must be initialized at track %d' % i

        for i,camera in enumerate(self.cameras):
            assert camera.R.shape == (3,3)
            assert camera.t.shape == (3,)

    # Add a new camera. If a camera is provided, initialize it with
    # that value. Otherwise, create a camera with default parameters
    # and add it. In either case, assign the camera the next ID and
    # return a reference to it.
    def add_camera(self, camera=None):
        if camera is None:
            camera = Camera(np.eye(3), np.zeros(3))
        camera.idx = len(self.cameras)
        self.cameras.append(camera)
        return camera

    # Add a new track. If a track is provided, initialize it with that
    # value. Otherwise, create an empty track and add it. In either
    # case, return a reference to the added track.
    def add_track(self, track=None):
        if track is None:
            track = Track()
        else:
            # Check camera ids
            for i,idx in enumerate(track.camera_ids()):
                if idx < 0 or idx >= len(self.cameras):
                    fmt = 'Invalid camera ID=%d in new track at camera_ids[%d])'
                    raise Exception(fmt % (int(idx), i))

        self.tracks.append(track)
        return track

    # Get a list of all rotation matrices
    def Rs(self):
        return np.array([ cam.R for cam in self.cameras ])

    # Get a list of all translation vectors
    def ts(self):
        return np.array([ cam.t for cam in self.cameras ])

    # Get a list of reconstructed points (one for each track)
    def points(self):
        return np.array([ track.reconstruction for track in self.tracks ])

    # Get the measurement of the j-th track in the i-th camera
    def measurement(self, i, j):
        return self.tracks[j].get_measurement(i)

    # Get (camer_id, track_id) pairs for all measurements
    def measurement_ids(self, track_indices=None):
        if track_indices is None:
            track_indices = xrange(len(self.tracks))
        return ( (i,j) for j in track_indices for i in self.tracks[j].camera_ids() )

    # Get (camer_id, track_id) pairs for all measurements
    def measurement_ids_for_cameras(self,
                                    cameras_to_include,
                                    track_indices=None):
        if track_indices is None:
            track_indices = xrange(len(self.tracks))
        return ( (i,j) 
                 for j in track_indices 
                 for i in self.tracks[j].intersect_camera_ids(cameras_to_include) )

    # Get the number of free parameters in the entire system
    def num_params(self):
        return len(self.cameras) * Bundle.NumCamParams + \
            len(self.tracks) * Bundle.NumPointParams

    # Get the prediction for the j-th track in the i-th camera
    def predict(self, i, j):
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

    # Get the complete residual vector.
    def residuals(self):
        return np.hstack([ self.residual(i,j) for (i,j) in self.measurement_ids() ])

    # Get the total cost of the system
    def complete_cost(self):
        return np.sum(np.square(self.residuals()))

    # Triangulate the position of a track. Returns a 3-vector
    def triangulate(self, track):
        Rs = [ self.cameras[idx].R for idx in track.camera_ids() ]
        ts = [ self.cameras[idx].t for idx in track.camera_ids() ]
        msms = track.measurements.values()
        return triangulate.algebraic_lsq(self.K, Rs, ts, msms)

    # Replace all tracks with their triangulation given current camera poses
    def triangulate_all(self):
        for track in self.tracks:
            track.reconstruction = self.triangulate(track)

    # Create a bundle from matrices of observations. For N cameras and M tracks:
    #  - K should 3 x 3
    #  - Rs should be N x 3 x 3
    #  - ts should be N x 3
    #  - measurements should be N x M x 2
    #  - measurement_mask should be N x M and of type bool
    # every False element of measurement_mask corresponds to a missing measurement
    @classmethod
    def FromArrays(cls, K, Rs, ts, pts, measurements, measurement_mask=None):
        K = np.asarray(K)
        Rs = np.asarray(Rs)
        ts = np.asarray(ts)
        measurements = np.asarray(measurements)

        if measurement_mask is None:
            measurement_mask = np.ones(measurements.shape[:-1], bool)

        # Check sizes
        assert len(Rs) == len(ts)
        assert np.shape(Rs)[1:] == (3,3)
        assert np.shape(ts)[1:] == (3,)
        assert measurements.shape[0] == len(Rs)
        assert measurements.shape[2] == 2
        assert measurement_mask.shape == measurements.shape[:-1]

        # Create the bundle
        b = Bundle()
        b.K = K
        for R,t in zip(Rs,ts):
            b.add_camera(Camera(R,t))
        for i in range(measurements.shape[1]):
            camera_ids = list(np.nonzero(measurement_mask[:,i])[0])
            msms = measurements[ camera_ids, i ]
            t = b.add_track(Track(camera_ids, msms))
            t.reconstruction = pts[i]
            
        # Return the bundle
        return b






    ################################################################################
    # These functions are now only used for unit-testing bundle adjustment. They are
    # not used "in production". They should be moved elsewhere
    ################################################################################

    # Apply a linear update to all cameras and points
    def perturb(self, delta, param_mask=None):
        delta = np.asarray(delta)
        nparams = self.num_params()

        if param_mask is not None:
            assert np.shape(param_mask) == (nparams,), \
                'shape was '+str(np.shape(param_mask))

            param_mask = np.asarray(param_mask)
            reduced_delta = delta.copy()
            delta = np.zeros(nparams)
            delta[param_mask] = reduced_delta

        assert delta.shape == (nparams,), 'shape was '+str(np.shape(delta))

        for i,cam in enumerate(self.cameras):
            cam.perturb( delta[ i*Bundle.NumCamParams : (i+1)*Bundle.NumCamParams ] )

        offs = len(self.cameras) * Bundle.NumCamParams
        for i,track in enumerate(self.tracks):
            track.perturb( delta[ offs+i*Bundle.NumPointParams : offs+(i+1)*Bundle.NumPointParams ] )

        return self


    # Get the complete residual vector. It is a vector of length
    # NUM_MEASUREMENTS*2 where NUM_MEASUREMENTS is the sum of the
    # track lengths. If tracks_to_include is not None then only get residuals
    # for those tracks. If cameras_to_include is not None then
    # restrict residuals to observations in cameras specified in that
    # vector.
    def residuals_partial(self, camera_ids, track_ids):
        nc = len(camera_ids)
        nt = len(track_ids)
        nparams = nc*6 + nt*3
        col_offs = nc*6

        # Build up the jacobian as a list of rows where each chunk is
        # a pair of rows corresponding to a particular measurement.
        chunks = []
        for jpos,j in enumerate(track_ids):
            track = self.tracks[j]
            for ipos,i in enumerate(camera_ids):
                if track.has_measurement(i):
                    chunks.append(self.residual(i,j))
        return np.hstack(chunks)

    # Get the Jacobian of the complete residual vector.
    def Jresiduals(self):
        return self.Jresiduals_extended()[0]

    # Get the Jacobian of the complete residual vector, together with
    # row and column labels
    def Jresiduals_extended(self):
        # TODO: do this the efficient way
        msm_ids = list(self.measurement_ids())
        J = np.zeros((len(msm_ids)*2, self.num_params()))
        row_labels = np.empty((len(msm_ids)*2, 2), int)
        col_labels = np.empty((self.num_params(), 2), int)
        col_offs = len(self.cameras) * Bundle.NumCamParams

        for msm_idx,(i,j) in enumerate(msm_ids):
            # Compute jacobians
            Jr_cam, Jr_x = self.Jresidual(i,j)

            # Slot into the main array
            row = msm_idx*2
            camcol = i * Bundle.NumCamParams
            ptcol = col_offs + j * Bundle.NumPointParams
            J[ row:row+2, camcol:camcol+Bundle.NumCamParams  ] = Jr_cam
            J[ row:row+2,  ptcol:ptcol+Bundle.NumPointParams ] = Jr_x

            row_labels[ row:row+2 ] = (i,j)
            col_labels[ camcol:camcol+Bundle.NumCamParams ] = (i, -1)
            col_labels[ ptcol:ptcol+Bundle.NumPointParams ] = (-1, j)

        return J, row_labels, col_labels

    # Get the Jacobian of the residual vector restricted to a set of cameras and tracks
    def Jresiduals_partial(self, camera_ids=None, track_ids=None):
        nc = len(camera_ids)
        nt = len(track_ids)
        nparams = nc*6 + nt*3
        col_offs = nc*6

        # Build up the jacobian as a list of rows where each chunk is
        # a pair of rows corresponding to a particular measurement.
        chunks = []
        for jpos,j in enumerate(track_ids):
            track = self.tracks[j]
            for ipos,i in enumerate(camera_ids):
                if track.has_measurement(i):
                    chunk = np.zeros((2, nparams))
                    camcol = ipos * Bundle.NumCamParams
                    ptcol = col_offs + jpos * Bundle.NumPointParams

                    Jr_cam, Jr_x = self.Jresidual(i,j)
                    chunk[ :, camcol:camcol+Bundle.NumCamParams  ] = Jr_cam
                    chunk[ :,  ptcol:ptcol+Bundle.NumPointParams ] = Jr_x
                    chunks.append(chunk)

        return np.vstack(chunks)
