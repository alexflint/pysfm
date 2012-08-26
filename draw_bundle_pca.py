from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt

import pca

def plot_points(xs, *args, **kwargs):
    xs = asarray(xs)
    plt.plot(xs[:,0], xs[:,1], *args, **kwargs)

def draw_pose(R, t, subspace, color):
    C = -dot(R.T, t)
    Cp = pca.project(C, subspace)
    for i in range(3):
        ep = pca.project(C + R[:,i], subspace)
        plot_points([ Cp, ep ], '-'+color)

def draw_point_cloud(xs, subspace):
    plot_points(pca.project(xs, subspace), '.b')

def draw_bundle(bundle, subspace, pose_colors=None):
    #mpl.rcParams['legend.fontsize'] = 10

    plt.figure()

    pts = bundle.points()
    assert pts.shape[1] == 3
    draw_point_cloud(pts, subspace)
    for i,camera in enumerate(bundle.cameras):
        color = pose_colors[i] if pose_colors is not None else 'r'
        draw_pose(camera.R, camera.t, subspace, color)

def compute_subspace(bundle):
    return pca.compute(bundle.ts(), 2)

if __name__ == '__main__':
    import sys
    import bundle_io

    bundle = bundle_io.load(sys.argv[1], sys.argv[2])
    for track in bundle.tracks:
        track.reprojection = bundle.triangulate(track)

    draw_bundle(bundle)

