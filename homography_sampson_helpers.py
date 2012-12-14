import numpy as np
import homography_sampson_jacobian

def to_vec9(m):
    m = np.asarray(m)
    assert m.size == 9
    v = homography_sampson_jacobian.Vector9()
    for i in range(9):
        v[i] = m.flat[i]
    return v

def to_numpy(m):
    assert isinstance(m, homography_sampson_jacobian.Vector9)
    return np.array([ m[i] for i in range(9) ])

def error(x0, x1, H):
    assert np.shape(x0) == (2,)
    assert np.shape(x1) == (2,)
    return homography_sampson_jacobian.error(x0[0], x0[1], x1[0], x1[1], to_vec9(H))

def jacobian(x0, x1, H):
    assert np.shape(x0) == (2,)
    assert np.shape(x1) == (2,)

    v = homography_sampson_jacobian.Vector9()
    homography_sampson_jacobian.jacobian(x0[0], x0[1], x1[0], x1[1], to_vec9(H), v)
    return to_numpy(v)
