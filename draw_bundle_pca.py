from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt

import pca

def plot_points(xs, *args, **kwargs):
    xs = asarray(xs)
    plt.plot(xs[:,0], xs[:,1], *args, **kwargs)

def draw_pose(R, t, subspace=None, color=None):
    if subspace is None:
        subspace = eye(3)[:2]
    if color is None:
        color = 'r'

    C = -dot(R.T, t)
    Cp = pca.project(C, subspace)
    for i in range(3):
        ep = pca.project(C + R[:,i], subspace)
        plot_points([ Cp, ep ], '-'+color)

def draw_poses(Rs, ts, subspace=None, color=None):
    plt.clf()
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    for R,t in zip(Rs,ts):
        draw_pose(R,t,subspace,color)

def draw_point_cloud(xs, subspace=None):
    if subspace is None:
        subspace = eye(3)[:2]
    plot_points(pca.project(xs, subspace), '.b')

def draw_bundle(bundle, subspace=None, pose_colors=None):
    if subspace is None:
        subspace = eye(3)[:2]
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
