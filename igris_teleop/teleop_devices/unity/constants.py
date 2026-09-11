import numpy as np


T_to_unitree_left_wrist = np.array([[ 0,  0, -1, 0],
                                        [-1,  0,  0, 0],
                                        [ 0,  1,  0, 0],
                                        [ 0,  0,  0, 1]])


# np.array([[1, 0, 0, 0],
#                                     [0, 0, -1, 0],
#                                     [0, 1, 0, 0],
#                                     [0, 0, 0, 1]])

T_to_unitree_right_wrist = np.array([[ 0,  0, -1, 0],
                                         [ 1,  0,  0, 0],
                                         [ 0, -1,  0, 0],
                                         [ 0,  0,  0, 1]])

# T_to_unitree_right_wrist = np.array([
#     [1, 0, 0, 0],   # x_U = x_XR  (wrist→middle)
#     [0, 0,-1, 0],   # y_U = -z_XR (back→palm)
#     [0, 1, 0, 0],   # z_U = y_XR  (pinky→index)
#     [0, 0, 0, 1],
# ])


# np.array([[1, 0, 0, 0],
#                                      [0, 0, 1, 0],
#                                      [0, -1, 0, 0],
#                                      [0, 0, 0, 1]])

T_robot_openxr = np.array([[0, 0, -1, 0],
                           [-1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]])

# Historical name used for OpenXR point arrays. Keep one canonical basis matrix.
grd_yup2grd_zup = T_robot_openxr


# hand2inspire = np.array([
#     [0, -1, 0, 0],
#     [0, 0, -1, 0],
#     [1, 0, 0, 0],
#     [0, 0, 0, 1]
# ])

# 손 좌표계를 y축 +90° 회전한 뒤, 그 좌표계 기준 z축으로 +90° 회전한 변환
righthand2igris = np.array([
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])

lefthand2igris = np.array([
    [ 0,  0,  1, 0],
    [-1,  0,  0, 0],
    [ 0, -1,  0, 0],
    [ 0,  0,  0, 1],
])


# np.array([
#     [0,  1,  0, 0],
#     [0,  0, -1, 0],
#     [-1, 0,  0, 0],
#     [0,  0,  0, 1],
# ])
###############
T_to_unitree_hand = np.array([[0, 0, 1, 0],
                              [-1,0, 0, 0],
                              [0, -1,0, 0],
                              [0, 0, 0, 1]])

const_head_vuer_mat = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 1.5],
                                [0, 0, 1, -0.2],
                                [0, 0, 0, 1]])


# For G1 initial position
# const_right_wrist_vuer_mat = np.array([[1, 0, 0, 0.15],
#                                        [0, 1, 0, 1.13],
#                                        [0, 0, 1, -0.3],
#                                        [0, 0, 0, 1]])

# # For G1 initial position
# const_left_wrist_vuer_mat = np.array([[1, 0, 0, -0.15],
#                                       [0, 1, 0, 1.13],
#                                       [0, 0, 1, -0.3],
#                                       [0, 0, 0, 1]])

const_right_wrist_vuer_mat = np.array([[1, 0, 0, 0.15],
                                       [0, 1, 0, 0.8],
                                       [0, 0, 1, -0.3],
                                       [0, 0, 0, 1]])

const_left_wrist_vuer_mat = np.array([[1, 0, 0, -0.15],
                                      [0, 1, 0, 0.8],
                                      [0, 0, 1, -0.3],
                                      [0, 0, 0, 1]])
