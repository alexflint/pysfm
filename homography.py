from numpy import *
import algebra

################################################################################
def estimate(fp, tp):
    fp = algebra.unpr(asarray(fp, float)).T
    tp = algebra.unpr(asarray(tp, float)).T
    return estimate_lsq(fp, tp)

################################################################################
def estimate_lsq(fp, tp):
    """ find homography H, such that fp is mapped to tp
        using the linear DLT method. Points are conditioned
        automatically."""

    fp = asarray(fp, float)
    tp = asarray(tp, float)

    if fp.shape[0] != 3:
        raise RuntimeError, 'number of rows in fp must be 3 (there were %d)' % fp.shape[0]

    if tp.shape[0] != 3:
        raise RuntimeError, 'number of rows in tp must be 3 (there were %d)' % tp.shape[0]

    if fp.shape[1] != tp.shape[1]:
        raise RuntimeError, 'number of points do not match'

    #condition points (important for numerical reasons)
    #--from points--
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1))
    if abs(maxstd) < 1e-8:
        # This is a degenerate configuration
        raise linalg.LinAlgError

    C1 = diag([1/maxstd, 1/maxstd, 1]) 
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = dot(C1,fp)

    #--to points--
    m = mean(tp[:2], axis=1)
    #C2 = C1.copy() #must use same scaling for both point sets
    maxstd = max(std(tp[:2], axis=1))
    if abs(maxstd) < 1e-8:
        # This is a degenerate configuration
        raise linalg.LinAlgError

    C2 = diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = dot(C2,tp)

    #create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = zeros((2*nbr_correspondences,9))
    for i in range(nbr_correspondences):        
        A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
        A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]

    U,S,V = linalg.svd(A)

    H = V[8].reshape((3,3))    

    #decondition and return
    return dot(linalg.inv(C2),dot(H,C1))
