from numpy import *
from algebra import *

# Get a relative pose (R01,t01) that goes from the pose (R0,t0) to (R1,t1)
def relative_pose(R0, t0, R1, t1):
    R_delta = dot(R1, R0.T)
    t_delta = t1 - dots(R1, R0.T, t0)
    return R_delta, t_delta


# Take the relative pose from (R0,t0) to (R0_updated,t0_updated)
# and apply it to (R1,t1) to get (R1_updated,t1_updated)
def propagate_pose_update(R0, t0, R0_updated, t0_updated, R1, t1):
    R_delta = dot(R0_updated, R0.T)
    R1_updated = dot(R_delta, R1)
    t1_updated = dot(R_delta, t1 - t0) + t0_updated
    return R1_updated, t1_updated

def propagate_pose_update_inplace(cam0, cam0_updated, cam1):
    R1_upd, t1_upd = propagate_pose_update(cam0.R, cam0.t,
                                           cam0_updated.R, cam0_updated.t,
                                           cam1.R, cam1.t)
    cam1.R = R1_upd
    cam1.t = t1_upd



def rotation_xy(th):
    return np.array([[ np.cos(th), -np.sin(th), 0 ],
                     [ np.sin(th), np.cos(th),  0 ],
                     [ 0,          0,           1 ]])

def rotation_xz(th):
    return np.array([[ np.cos(th),  0,  -np.sin(th) ],
                     [ 0,           1,  0           ],
                     [ np.sin(th),  0,  np.cos(th), ]])
