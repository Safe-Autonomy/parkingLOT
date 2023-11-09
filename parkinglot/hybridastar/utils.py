import numpy as np
from scipy.spatial.transform import Rotation as Rot

def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle
    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]