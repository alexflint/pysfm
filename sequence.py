import numpy as np
from numpy import *

import bundle
import triangulate

# Represents a observations of 3D point from a sequence of cameras
class Track:
    def __init__(self, camera_ids, measurements):
        camera_ids = np.asarray(camera_ids)
        measurements = np.asarray(measurements)
        assert len(camera_ids) == len(measurements)
        self.camera_ids = camera_ids
        self.measurements = measurements

    def __str__(self):
        s = ''
        for cam_id,msm in zip(self.camera_ids, self.measurements):
            s += 'Camera %d: (%f,%f)\n' % (cam_id, msm[0], msm[1])
        return s

    def __repr__(self):
        return str(self)

# Represents a set of observations over an image sequence
class Sequence(object):
    def __init__(self):
        self.tracks = []
        self.Rs = []
        self.ts = []
        self.xs = []
        self.K = np.eye(3)

    # Pull out a window of n frames. Create a Bundle object containing
    # measurements for all tracks in those frames
    def get_window(self, first, n):
        assert len(self.tracks) == len(self.xs)
        assert len(self.Rs)     == len(self.ts)
        
        # Get a vector of measurements for this track + window
        track_ids = []
        window = []
        window_pts = []
        for track,pt in zip(self.tracks, self.xs):
            msm_vec = -np.ones((n, 2))
            observed = 0
            for cam_id,msm in zip(track.camera_ids, track.measurements):
                if cam_id >= first and cam_id < first+n:
                    msm_vec[ cam_id-first ] = msm
                    observed += 1
            if observed >= 2:
                window.append(msm_vec)
                window_pts.append(pt)    # TODO: deal with missing data

        # Count observations per camera
        window = np.asarray(window)
        obs_mask = np.all(window != (-1,-1), axis=2)
        obs_per_cam = obs_mask.sum(axis=0)

        # Check for cameras with too few observations
        deg_cams = np.nonzero(obs_per_cam < 3)[0]
        assert len(deg_cams) == 0, 'Some cameras had too few observations: '+str(obs_per_cam)

        # Construct a bundle instance
        b = bundle.Bundle(n, len(window))
        b.K = self.K
        b.msm = np.transpose(window, (1,0,2))  # swap first two dimensions
        b.Rs = [ self.Rs[i] for i in range(first, first+n) ]
        b.ts = [ self.ts[i] for i in range(first, first+n) ]
        b.pts = window_pts

        return b

    # Triangulate the position of the point corresponding to the i-th track
    def triangulate(self, i):
        track = self.tracks[i]
        Rs = [ self.Rs[j] for j in track.camera_ids ]
        ts = [ self.ts[j] for j in track.camera_ids ]
        return triangulate.triangulate(self.K, Rs, ts, track.measurements)

    def write(self, track_fd, cams_fd):
        # Write tracks
        for track in self.tracks:
            for idx,msm in zip(track.camera_ids, track.measurements):
                track_fd.write('%d %f %f ' % (idx, msm[0], msm[1]))
            track_fd.write('\n')

        # Write initial cameras
        for row in self.K:
            cams_fd.write('%f %f %f\n' % tuple(row))
        for R,t in zip(self.Rs, self.ts):
            P = np.hstack((R, t[:,np.newaxis]))
            for row in P:
                cams_fd.write('%f %f %f %f\n' % tuple(row))

    @classmethod
    def load(cls, track_fd, cams_fd):
        seq = Sequence()

        # Load tracks
        for line in track_fd:
            tokens = line.strip().split(' ')
            cam_ids = map(int, tokens[0::3])
            msm_xs = map(float, tokens[1::3])
            msm_ys = map(float, tokens[2::3])
            seq.tracks.append(Track(cam_ids, zip(msm_xs, msm_ys)))

        # Load initial cameras
        tokens = cams_fd.read().strip().split()
        vs = np.array(map(float, tokens))   # get entire file as a list of floats
        seq.K = vs[:9].reshape((3,3))
        Ps = vs[9:].reshape((-1, 4))  # -1 means that dimension size is inferred
        seq.Rs = list(Ps[:,:3].reshape(-1, 3, 3)) # divide cols 1..3 into chunks of 3
        seq.ts = list(Ps[:,3].reshape(-1, 3))     # divide cols 4 into chunks of 3

        # Triangulate initial points
        seq.xs = [ seq.triangulate(i) for i in range(len(seq.tracks)) ]

        return seq
