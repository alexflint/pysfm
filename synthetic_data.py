import os
import sys
import numpy as np
from numpy import *
from lie import SO3
from algebra import *

import bundle
import sequence


# generate a set of measurements with Gaussian noise
def generate_measurements(K, R, t, pts, noise):
    assert pts.ndim == 2, 'pts.shape = '+str(pts.shape)
    assert pts.shape[1] == 3, 'pts.shape = '+str(pts.shape)
    return np.array([ bundle.project(K, R, t, x) + np.random.randn(2)*noise
                      for x in pts ])

# Sample a 3x3 rotation
def generate_rotation(noise):
    return SO3.exp(np.random.randn(3) * noise)

def perturb_rotation(R, noise):
    return dots(R, SO3.exp(np.random.randn(3) * noise))

def perturb_vector(v, noise):
    return v + np.random.randn(*v.shape) * noise

# Generate a sequence of tracks
def generate_sequence(nframes, npts, msm_noise=.02):
    np.random.seed(654)  # repeatability

    pt_radius = 1.
    R_pert = .02
    t_pert = .02

    # Generate points
    pts = (np.random.rand(npts,3) * 2 - 1) * pt_radius

    # Generate cameras
    K = np.eye(3)
    R = np.eye(3)
    t = np.zeros(3)
    Rs = []
    ts = []
    measurements = []
    for i in range(nframes):
        Rs.append(R)
        ts.append(t)
        measurements.append(generate_measurements(K, R, t, pts, msm_noise))
        R = perturb_rotation(Rs[-1], R_pert)
        t = perturb_vector(ts[-1], t_pert)

    measurements = np.array(measurements)

    # Convert into tracks
    tracks = []
    for track_msms in np.transpose(measurements, (1,0,2)):
        tracks.append(sequence.Track(arange(nframes), track_msms))
        
    # Create the sequence
    seq = sequence.Sequence()
    seq.K = K
    seq.tracks = tracks
    seq.initial_Rs = Rs  # TODO: add noise
    seq.initial_ts = ts  # TODO: add noise
    seq.initial_xs = pts # TODO: add noise

    return seq

def create_test_sequence():
    return generate_sequence(10, 6)

def check_window_extraction():
    win_offs = 1
    seq = create_test_sequence()
    win = seq.get_window(win_offs, 4)

    for i in range(4):
        assert np.all(win.Rs[i] == seq.initial_Rs[i+win_offs]), \
            'win.Rs = %s\n, initial_Rs=%s' % (str(win.Rs[i]), str(seq.initial_Rs[i+win_offs]))
        assert np.all(win.ts[i] == seq.initial_ts[i+win_offs]), \
            'win.ts = %s\n, initial_ts=%s' % (str(win.ts[i]), str(seq.initial_ts[i+win_offs]))
        for j in range(6):
            assert np.all(win.msm[i,j] == seq.tracks[j].measurements[i+win_offs]), \
                'at cam %d, track %d: %s vs %s' % \
                (i, j, str(win.msm[i,j]), str(seq.tracks[j].measurements[i+win_offs]))

    print 'Window test passed'

if __name__ == '__main__':
    check_window_extraction()
    
    seq = create_test_sequence()
    with open('tracks.txt', 'w') as tracks_fd:
        with open('cameras.txt', 'w') as cams_fd:
            seq.write(tracks_fd, cams_fd)
