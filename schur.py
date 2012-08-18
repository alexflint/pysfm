import numpy as np
from algebra import *

def get_schur_complement(A, b, n):
    A = np.asarray(A)
    m = A.shape[0] - n

    V = dots(A[:n, -m:], np.linalg.inv(A[-m:,-m:]))
    Acomp = A[:n, :n] - dots(V, A[-m:, :n])
    bcomp = b[:n]     - dots(V, b[-m:])
    return Acomp,bcomp

def solve_by_schur(A, b, n, mask=None):
    A = np.asarray(A)
    assert A.shape[0] == A.shape[1], 'A should be square'
    assert n < A.shape[0], 'n=%d, A.shape[0]=%d' % (n, A.shape[0])
    if mask is not None:
        mask = mask[:n]

    m = A.shape[0] - n

    # Get first linear system
    S, b1 = get_schur_complement(A, b, n)
    if mask is not None:
        S = S[mask].T[mask].T
        b1 = b1[mask]

    # Solve for top part of x
    x1 = np.linalg.solve(S, b1)

    # Put zeros in place of the parameters we didn't solve for
    # This really is equivalent to the "thing we really want to do"
    if mask is not None:
        x1_reduced = x1
        x1 = np.zeros(n)
        x1[mask] = x1_reduced

    # Backsubstitute
    C = A[-m:, :n]
    D = A[-m:, -m:]
    b2 = b[-m:] - dots(C, x1)
    x2 = np.linalg.solve(D, b2)

    return np.concatenate((x1, x2))

if __name__ == '__main__':
    A = np.array([[1,2,3,4],
                  [5,4,3,2],
                  [8,7,2,2],
                  [9,9,9,10]], float)
    print np.linalg.det(A)
    
    b = np.arange(4)
    print np.linalg.solve(A, b)

    print solve_by_schur(A, b, 1)
    print solve_by_schur(A, b, 2)
    print solve_by_schur(A, b, 3)
