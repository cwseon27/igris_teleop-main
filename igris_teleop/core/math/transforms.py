import numpy as np


def common_frame_relative_pose(current, reference):
    """Return the world-axis motion between poses expressed in one common frame."""
    current = np.asarray(current, dtype=np.float64).reshape(4, 4)
    reference = np.asarray(reference, dtype=np.float64).reshape(4, 4)

    relative = np.eye(4, dtype=np.float64)
    relative[:3, :3] = current[:3, :3] @ reference[:3, :3].T
    relative[:3, 3] = current[:3, 3] - reference[:3, 3]
    return relative


def apply_common_frame_relative_pose(relative, anchor):
    """Apply world-axis relative motion to a destination anchor pose."""
    relative = np.asarray(relative, dtype=np.float64).reshape(4, 4)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(4, 4)

    target = anchor.copy()
    target[:3, :3] = relative[:3, :3] @ anchor[:3, :3]
    target[:3, 3] = anchor[:3, 3] + relative[:3, 3]
    return target


def mat_update(prev_mat, mat):
    if np.linalg.det(mat) == 0:
        return prev_mat, False # Return previous matrix and False flag if the new matrix is non-singular (determinant ≠ 0).
    else:
        return mat, True


def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret
