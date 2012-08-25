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
    camera_data = np.loadtxt(open(cameras_path))
    for row in camera_data:
        P = row.reshape((3,4))
        bundle.add_camera(Camera(P[:,:3], P[:,3]))

    # Read tracks
    for i,line in enumerate(open(tracks_path)):
        tokens = line.strip().split()
        assert len(tokens) % 3 == 0, 'Error at line %d:\n %s' % (i,line)
        vs = np.array(map(int, tokens)).reshape((-1,3))
        bundle.add_track(Track(list(vs[:,0].astype(int)), vs[:,1:]))

    return bundle
