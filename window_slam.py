import sys
import numpy as np
from numpy import *

from bundle_adjuster import BundleAdjuster
import bundle
import bundle_io
import sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from draw_bundle_3d import draw_bundle

class SlidingWindowSLAM(object):
    def __init__(self, window_size):
        self.window_size = window_size
        
    def run(self, measurements, init_poses):
        nf = len(sequence.Rs)
        for t in range(nf-self.window_size+1):
            b = sequence.get_window(t, self.window_size)


def run(complete_bundle, window_size, num_to_freeze=2, pdf_pattern=None):
    
    # Create binary mask for frozen cameras
    camera_mask = arange(window_size) < num_to_freeze

    num_tracks = 100
    track_ids = range(num_tracks)
    
    # Start optimizing
    cur_bundle = complete_bundle    
    for i in range(0, len(complete_bundle.cameras)-window_size):
        print '\n\n==============\nWINDOW: [%d..%d]\n' % (i, i+window_size)

        # Adjust this window
        camera_ids = range(i, i+window_size)
        ba = BundleAdjuster()
        ba.set_bundle(cur_bundle, camera_ids=camera_ids, track_ids=track_ids)
        ba.optimize()
        
        # Save the window
        cur_bundle = ba.bundle
        if pdf_pattern is not None:
            pdf_path = pdf_pattern % i
            print 'Writing to ',pdf_path
            draw_bundle(cur_bundle)
            plt.savefig(pdf_path)


if __name__ == '__main__':
    np.seterr(all='raise')

    if len(sys.argv) > 4:
        pdf_pattern = sys.argv[4]
    else:
        pdf_pattern = None

    bundle = bundle_io.load(sys.argv[1], sys.argv[2])
    bundle.triangulate_all()

    print pdf_pattern

    print 'Cameras:',len(bundle.cameras)
    print 'Tracks:',len(bundle.tracks)
    run(bundle, int(sys.argv[3]), pdf_pattern=pdf_pattern)

