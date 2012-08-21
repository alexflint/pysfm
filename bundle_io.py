import numpy as np
from bundle import Bundle, Camera, Track

# Eek, currently hardcoded
width = 1480;
height = 1360;
K = np.array([ 1500, 0, width/2, 0, 1500, height/2, 0, 0, 1 ], float).reshape((3,3))

def load(tracks_path, cameras_path):
    bundle = Bundle()
    bundle.K = K

    # Read cameras
    for i,line in enumerate(open(cameras_path)):
        tokens = line.strip().split()
        assert len(tokens) == 12, 'Error at line %d:\n %s' % (i,line)
        P = np.array(map(float, tokens)).reshape((3,4))
        bundle.add_camera(Camera(P[:,:3], P[:,3]))

    # Read tracks
    for i,line in enumerate(open(tracks_path)):
        tokens = line.strip().split()
        assert len(tokens) % 3 == 0, 'Error at line %d:\n %s' % (i,line)
        vs = np.array(map(int, tokens)).reshape((-1,3))
        t = bundle.add_track(Track(list(vs[:,0]), vs[:,1:]))

    return bundle
