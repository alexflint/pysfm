import sys
import numpy as np
import unittest

def spy(A, t=1e-8):
    A = np.asarray(A)
    assert A.ndim == 2
    for row in A:
        sys.stdout.write('[')
        sys.stdout.write(np.take([' ','x'], np.abs(row)>t))
        sys.stdout.write(']\n')

############################################################################
# Helpers for numpy testing
class NumpyTestCase(unittest.TestCase):
    def assertShape(self, arr, expected_shape):
        self.assertEqual(np.shape(arr), expected_shape)
