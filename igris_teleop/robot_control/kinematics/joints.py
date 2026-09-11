from enum import IntEnum


NUM_MOTORS = 31


class JointIndex(IntEnum):
    """PJS (joint/PR space) semantic names matching LowCmd.motors() ordering."""

    WAIST_YAW = 0
    WAIST_ROLL = 1
    WAIST_PITCH = 2

    L_HIP_PITCH = 3
    L_HIP_ROLL = 4
    L_HIP_YAW = 5
    L_KNEE_PITCH = 6
    L_ANKLE_PITCH = 7
    L_ANKLE_ROLL = 8

    R_HIP_PITCH = 9
    R_HIP_ROLL = 10
    R_HIP_YAW = 11
    R_KNEE_PITCH = 12
    R_ANKLE_PITCH = 13
    R_ANKLE_ROLL = 14

    L_SHOULDER_PITCH = 15
    L_SHOULDER_ROLL = 16
    L_SHOULDER_YAW = 17
    L_ELBOW_PITCH = 18
    L_WRIST_YAW = 19
    L_WRIST_ROLL = 20
    L_WRIST_PITCH = 21

    R_SHOULDER_PITCH = 22
    R_SHOULDER_ROLL = 23
    R_SHOULDER_YAW = 24
    R_ELBOW_PITCH = 25
    R_WRIST_YAW = 26
    R_WRIST_ROLL = 27
    R_WRIST_PITCH = 28

    NECK_YAW = 29
    NECK_PITCH = 30


class MotorIndex(IntEnum):
    """MS (motor/AB space) semantic names matching LowCmd.motors() ordering."""

    WAIST_YAW = 0
    WAIST_L = 1
    WAIST_R = 2

    HIP_PITCH_L = 3
    HIP_ROLL_L = 4
    HIP_YAW_L = 5
    KNEE_PITCH_L = 6
    ANKLE_OUT_L = 7
    ANKLE_IN_L = 8

    HIP_PITCH_R = 9
    HIP_ROLL_R = 10
    HIP_YAW_R = 11
    KNEE_PITCH_R = 12
    ANKLE_OUT_R = 13
    ANKLE_IN_R = 14

    SHOULDER_PITCH_L = 15
    SHOULDER_ROLL_L = 16
    SHOULDER_YAW_L = 17
    ELBOW_PITCH_L = 18
    WRIST_YAW_L = 19
    WRIST_FRONT_L = 20
    WRIST_BACK_L = 21

    SHOULDER_PITCH_R = 22
    SHOULDER_ROLL_R = 23
    SHOULDER_YAW_R = 24
    ELBOW_PITCH_R = 25
    WRIST_YAW_R = 26
    WRIST_FRONT_R = 27
    WRIST_BACK_R = 28

    NECK_YAW = 29
    NECK_PITCH = 30


LEG_INDICES = (
    JointIndex.L_HIP_PITCH,
    JointIndex.L_HIP_ROLL,
    JointIndex.L_HIP_YAW,
    JointIndex.L_KNEE_PITCH,
    JointIndex.L_ANKLE_PITCH,
    JointIndex.L_ANKLE_ROLL,
    JointIndex.R_HIP_PITCH,
    JointIndex.R_HIP_ROLL,
    JointIndex.R_HIP_YAW,
    JointIndex.R_KNEE_PITCH,
    JointIndex.R_ANKLE_PITCH,
    JointIndex.R_ANKLE_ROLL,
)

WAIST_INDICES = (
    JointIndex.WAIST_YAW,
    JointIndex.WAIST_ROLL,
    JointIndex.WAIST_PITCH,
)

L_SHOULDER_INDICES = (
    JointIndex.L_SHOULDER_PITCH,
    JointIndex.L_SHOULDER_ROLL,
    JointIndex.L_SHOULDER_YAW,
)

R_SHOULDER_INDICES = (
    JointIndex.R_SHOULDER_PITCH,
    JointIndex.R_SHOULDER_ROLL,
    JointIndex.R_SHOULDER_YAW,
)

L_ELBOW_INDICES = (
    JointIndex.L_ELBOW_PITCH,
)

R_ELBOW_INDICES = (
    JointIndex.R_ELBOW_PITCH,
)

L_WRIST_INDICES = (
    JointIndex.L_WRIST_YAW,
    JointIndex.L_WRIST_ROLL,
    JointIndex.L_WRIST_PITCH,
)

R_WRIST_INDICES = (
    JointIndex.R_WRIST_YAW,
    JointIndex.R_WRIST_ROLL,
    JointIndex.R_WRIST_PITCH,
)

L_ARM_INDICES = (
    *L_SHOULDER_INDICES,
    *L_ELBOW_INDICES,
    *L_WRIST_INDICES,
)

R_ARM_INDICES = (
    *R_SHOULDER_INDICES,
    *R_ELBOW_INDICES,
    *R_WRIST_INDICES,
)

ARM_INDICES = (
    *L_ARM_INDICES,
    *R_ARM_INDICES,
)

UPPER_ARM_INDICES = (
    *L_SHOULDER_INDICES,
    *L_ELBOW_INDICES,
    *R_SHOULDER_INDICES,
    *R_ELBOW_INDICES,
)

HAND_ARM_INDICES = (
    *L_WRIST_INDICES,
    *R_WRIST_INDICES,
)

NECK_INDICES = (
    JointIndex.NECK_YAW,
    JointIndex.NECK_PITCH,
)
