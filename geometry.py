from numpy import *
from algebra import *

# Take the relative pose from (R0,t0) to (R0_updated,t0_updated)
# and apply it to (R1,t1) to get (R1_updated,t1_updated)
def propagate_pose_update(R0, t0, R0_updated, t0_updated, R1, t1):
    R_delta = dot(R0_updated, R0.T)
    R1_updated = dot(R_delta, R1)
    t1_updated = dot(R_delta, t1 - t0) + t0_updated
    return (R1_updated, t1_updated)


