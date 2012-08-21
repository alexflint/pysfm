import sys

import numpy as np
from copy import deepcopy

from algebra import *
from bundle import Bundle, Camera, Track, project
from bundle_adjuster import BundleAdjuster
import draw_bundle
import bundle_io

if __name__ == '__main__':
    tracks_path = sys.argv[1]
    cameras_path = sys.argv[2]

    bundle = bundle_io.load(tracks_path, cameras_path)

    for track in bundle.tracks:
        track.reconstruction = bundle.triangulate(track)

    NUM_CAMERAS = 100
    NUM_TRACKS = 100

    bundle.cameras = bundle.cameras[:NUM_CAMERAS]
    bundle.tracks = bundle.tracks[:NUM_TRACKS]
    for j,track in enumerate(bundle.tracks):
        track.measurements = { i : track.get_measurement(i) for i in range(NUM_CAMERAS) }

    param_mask = np.ones(bundle.num_params(), bool)
    param_mask[:6] = False
    param_mask[9] = False

    b_init = deepcopy(bundle)
    ba = BundleAdjuster(bundle)
    ba.optimize(param_mask, max_steps=10)

    # Write results
    with open('out/adjusted_poses.txt', 'w') as fd:
        for camera in ba.bundle.cameras:
            P = camera.projection_matrix()
            fd.write(''.join(['%f '%v for v in P.flatten()]))
            fd.write('\n')
    

    #draw_bundle.output_views(b_init, 'out/init.pdf')
    #draw_bundle.output_views(ba.bundle, 'out/estimated.pdf')
