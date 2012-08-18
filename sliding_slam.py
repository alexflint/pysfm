import bundle
import sequence

class SlidingWindowSLAM(object):
    def __init__(self, window_size):
        self.window_size = window_size
        
    def run(self, sequence):
        nf = len(sequence.Rs)
        for t in range(nf-self.window_size+1):
            b = sequence.get_window(t, self.window_size)
            
