import sys
import numpy as np
from numpy import *

from bundle_adjuster import BundleAdjuster
import bundle
import bundle_io
import sequence

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

from draw_bundle_pca import draw_bundle, compute_subspace

class SlidingWindowSLAM(object):
    def __init__(self, window_size):
        self.window_size = window_size
        
    def run(self, measurements, init_poses):
        nf = len(sequence.Rs)
        for t in range(nf-self.window_size+1):
            b = sequence.get_window(t, self.window_size)


def run(complete_bundle, window_size, num_to_freeze=2, pdf_pattern=None):
    NUM_TRACKS = 100

    # Determine the projection so that we always draw the bundle on a fixed subspace
    visualization_subspace = compute_subspace(complete_bundle)
    
    # Create binary mask for frozen cameras
    camera_mask = arange(window_size) < num_to_freeze

    track_ids = range(NUM_TRACKS)
    
    # Start optimizing
    cur_bundle = complete_bundle
    for i in range(0, len(complete_bundle.cameras)-window_size+1):
        print '\n\n==============\nWINDOW: [%d..%d]\n' % (i, i+window_size)

        # Adjust this window
        camera_ids = range(i, i+window_size)
        ba = BundleAdjuster()
        ba.set_bundle(cur_bundle, camera_ids=camera_ids, track_ids=track_ids)
        ba.optimize()
        cur_bundle = ba.bundle
        
        # Draw this window
        print 'Drawing bundle...'
        if pdf_pattern is not None:
            pose_colors = ['b'] * len(ba.bundle.cameras)
            for j in range(i):
                pose_colors[j] = 'g'
            for j in range(i,i+window_size):
                pose_colors[j] = 'r'
            pdf_path = pdf_pattern % i
            print 'Writing to ',pdf_path
            draw_bundle(ba.bundle, visualization_subspace, pose_colors)
            plt.savefig(pdf_path)

        break

if __name__ == '__main__':
    np.seterr(all='raise')

    if len(sys.argv) > 4:
        pdf_pattern = sys.argv[4]
    else:
        pdf_pattern = None

    window_size = int(sys.argv[3])

    print 'Loading bundle...'
    bundle = bundle_io.load(sys.argv[1], sys.argv[2])
    print 'Triangulating initial points...'
    bundle.triangulate_all()

    print 'Cameras:',len(bundle.cameras)
    print 'Tracks:',len(bundle.tracks)
    print 'Window Size:',window_size
    run(bundle, window_size, pdf_pattern=pdf_pattern)
