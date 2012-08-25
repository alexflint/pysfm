import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from numpy import *

import matplotlib.pyplot as plt

def stdunit(i,n=3):
    x = zeros(3)
    x[i] = 1
    return x

def plot_points(ax, xs, *args, **kwargs):
    xs = np.asarray(xs)
    ax.plot(xs[:,0], xs[:,1], xs[:,2], *args, **kwargs)

def draw_pose(ax, R, t):
    C = -dot(R.T, t)
    for i in range(3):
        ei = C + R[:,i]
        plot_points(ax, [ C, ei ], '-r')

def draw_point_cloud(ax, xs):
    plot_points(ax, xs, '.b')


def draw_bundle(bundle):
    mpl.rcParams['legend.fontsize'] = 10
    pts = np.random.randn(10,3).astype(float)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    draw_point_cloud(ax, bundle.points())
    for camera in bundle.cameras:
        draw_pose(ax, camera.R, camera.t)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    plt.show()


if __name__ == '__main__':
    import sys
    import bundle_io

    bundle = bundle_io.load(sys.argv[1], sys.argv[2])
    for track in bundle.tracks:
        track.reprojection = bundle.triangulate(track)

    draw_bundle(bundle)

