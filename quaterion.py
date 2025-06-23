import numpy as np
import mujoco


def create_rotation_quat(axis, angle):
    quat = np.zeros(4)
    angle = np.radians(angle)
    axis = np.array(axis)
    mujoco.mju_axisAngle2Quat(quat, axis, angle)
    return quat


def mul_quat(lh_quat, rh_quat):
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, lh_quat, rh_quat)
    return result


def create_rotation_quat(axis, angle):
    quat = np.zeros(4)
    angle = np.radians(angle)
    axis = np.array(axis)
    mujoco.mju_axisAngle2Quat(quat, axis, angle)
    return quat


def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, q1, q2)
    return result


# Basic rotation functions
def rot_x(degrees):
    """Rotate around X-axis"""
    return create_rotation_quat([1, 0, 0], degrees)


def rot_y(degrees):
    """Rotate around Y-axis"""
    return create_rotation_quat([0, 1, 0], degrees)


def rot_z(degrees):
    """Rotate around Z-axis"""
    return create_rotation_quat([0, 0, 1], degrees)


# Axis flipping functions
def flip_x():
    """Flip X-axis (180° rotation around Y-Z plane)"""
    return create_rotation_quat([0, 1, 0], 180)


def flip_y():
    """Flip Y-axis (180° rotation around X-Z plane)"""
    return create_rotation_quat([1, 0, 0], 180)


def flip_z():
    """Flip Z-axis (180° rotation around X-Y plane)"""
    return create_rotation_quat([0, 0, 1], 180)


# Composition function
def compose(*quaternions):
    """Compose multiple quaternions from left to right"""
    result = quaternions[0]
    for q in quaternions[1:]:
        result = quaternion_multiply(result, q)
    return result


# Common named rotations
def roll(degrees):
    """Pitch rotation (around X-axis)"""
    return rot_x(degrees)


def yaw(degrees):
    """Yaw rotation (around Z-axis)"""
    return rot_z(degrees)


def pitch(degrees):
    """Roll rotation (around Y-axis)"""
    return rot_y(degrees)


# Axis alignment functions
def align_x_to_y():
    """Rotate so X-axis points in Y direction"""
    return rot_z(90)


def align_x_to_z():
    """Rotate so X-axis points in Z direction"""
    return rot_y(-90)


def align_y_to_x():
    """Rotate so Y-axis points in X direction"""
    return rot_z(-90)


def align_y_to_z():
    """Rotate so Y-axis points in Z direction"""
    return rot_x(90)


def align_z_to_x():
    """Rotate so Z-axis points in X direction"""
    return rot_y(90)


def align_z_to_y():
    """Rotate so Z-axis points in Y direction"""
    return rot_x(-90)


# Utility for clean output
def quat_string(q, precision=2):
    """Format quaternion as clean string"""

    def format_component(val):
        if abs(val - round(val)) < 1e-10:
            return str(int(round(val)))
        else:
            return f"{val:.{precision}f}".rstrip('0').rstrip('.')

    return " ".join(format_component(comp) for comp in q)