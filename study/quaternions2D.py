import matplotlib.pyplot as plt
import numpy as np
import mujoco


def create_rotation_quat(axis, angle):
    quat = np.zeros(4)
    angle = np.radians(angle)
    axis = np.array(axis)
    mujoco.mju_axisAngle2Quat(quat, axis, angle)
    return quat


def quaternion(axis, theta_degrees):
    """
    Create a quaternion from rotation axis and angle in degrees using MuJoCo

    Parameters:
    axis: str - 'X', 'Y', or 'Z' (case insensitive)
    theta_degrees: float - rotation angle in degrees

    Returns:
    numpy array [w, x, y, z] - quaternion from MuJoCo
    """
    axis = axis.upper()
    if axis == 'X':
        axis_vec = [1, 0, 0]
    elif axis == 'Y':
        axis_vec = [0, 1, 0]
    elif axis == 'Z':
        axis_vec = [0, 0, 1]
    else:
        raise ValueError("Axis must be 'X', 'Y', or 'Z'")

    return create_rotation_quat(axis_vec, theta_degrees)


def quaternion_to_string(q, precision=3):
    """
    Convert quaternion array to string format "w x y z"

    Parameters:
    q: numpy array [w, x, y, z] - quaternion
    precision: int - number of decimal places (default: 3)

    Returns:
    str - formatted quaternion string
    """
    format_str = f"{{:.{precision}f}}"
    return f"{format_str.format(q[0])} {format_str.format(q[1])} {format_str.format(q[2])} {format_str.format(q[3])}"


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to 3x3 rotation matrix using MuJoCo"""
    # MuJoCo expects a flattened 9-element array for the rotation matrix
    R_flat = np.zeros(9)
    # Ensure quaternion is the right shape
    q_reshaped = q.reshape(-1)

    mujoco.mju_quat2Mat(R_flat, q_reshaped)

    # Reshape back to 3x3 matrix
    R = R_flat.reshape(3, 3)
    return R


def visualize_quaternion_2d(axis, theta_degrees):
    """Create 2D visualization of quaternion rotation"""
    # Get quaternion using MuJoCo
    q = quaternion(axis, theta_degrees)
    q_str = quaternion_to_string(q)

    # Get rotation matrix using MuJoCo
    R = quaternion_to_rotation_matrix(q)

    # Original coordinate frame vectors
    x_axis = np.array([1, 0, 0])  # Blue arrow (+X)
    y_axis = np.array([0, 1, 0])  # Red arrow (+Y)

    # Rotate the vectors
    x_rotated = R @ x_axis
    y_rotated = R @ y_axis

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Set up the coordinate system
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

    # Draw original (faded) coordinate frame
    ax.arrow(0, 0, 1, 0, head_width=0.05, head_length=0.05,
             fc='lightblue', ec='lightblue', alpha=0.3, linewidth=2)
    ax.arrow(0, 0, 0, 1, head_width=0.05, head_length=0.05,
             fc='lightcoral', ec='lightcoral', alpha=0.3, linewidth=2)

    # Draw rotated coordinate frame
    ax.arrow(0, 0, x_rotated[0], x_rotated[1], head_width=0.05, head_length=0.05,
             fc='blue', ec='blue', linewidth=3, label='+X axis')
    ax.arrow(0, 0, y_rotated[0], y_rotated[1], head_width=0.05, head_length=0.05,
             fc='red', ec='red', linewidth=3, label='+Y axis')

    # Add labels and title
    ax.set_xlabel('X (left-right)')
    ax.set_ylabel('Y (up-down)')
    ax.set_title(f'2D Quaternion Visualization (MuJoCo)\nRotation: {theta_degrees}° around {axis}-axis')

    # Display quaternion values and rotation info
    text_info = f'quaternion(axis="{axis}", theta={theta_degrees}°)\n\n'
    text_info += f'MuJoCo Quaternion: {q_str}\n'
    text_info += f'|q| = {np.linalg.norm(q):.3f}'

    ax.text(1.1, 1.1, text_info, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    ax.legend(loc='lower right')
    plt.tight_layout()
    return fig, ax


def demo_common_rotations():
    """Demonstrate common 2D rotations using MuJoCo quaternions"""
    rotations = [
        ('Z', 0), ('Z', 30), ('Z', 45),
        ('Z', 60), ('Z', 90), ('Z', 120),
        ('Z', 135), ('Z', 180), ('Z', 270)
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, (axis, angle) in enumerate(rotations):
        q = quaternion(axis, angle)
        q_str = quaternion_to_string(q, precision=2)
        R = quaternion_to_rotation_matrix(q)

        x_axis_vec = np.array([1, 0, 0])
        y_axis_vec = np.array([0, 1, 0])
        x_rotated = R @ x_axis_vec
        y_rotated = R @ y_axis_vec

        ax = axes[i]
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

        # Faded original frame
        ax.arrow(0, 0, 1, 0, head_width=0.03, head_length=0.03,
                 fc='lightblue', ec='lightblue', alpha=0.3)
        ax.arrow(0, 0, 0, 1, head_width=0.03, head_length=0.03,
                 fc='lightcoral', ec='lightcoral', alpha=0.3)

        # Rotated frame
        ax.arrow(0, 0, x_rotated[0], x_rotated[1], head_width=0.03, head_length=0.03,
                 fc='blue', ec='blue', linewidth=2)
        ax.arrow(0, 0, y_rotated[0], y_rotated[1], head_width=0.03, head_length=0.03,
                 fc='red', ec='red', linewidth=2)

        ax.set_title(f'quaternion({axis}, {angle}°)\n{q_str}', fontsize=9)

    plt.suptitle('Common 2D Rotations - MuJoCo Quaternions', fontsize=14, y=0.98)
    plt.tight_layout()
    return fig


def print_mujoco_quaternion_table():
    """Print a table of common MuJoCo quaternions as strings"""
    print("Common 2D Quaternions (Z-axis rotations) - MuJoCo Format:")
    print("=" * 60)
    print(f"{'Angle':<8} {'Function Call':<25} {'MuJoCo Quaternion String'}")
    print("-" * 60)

    angles = [0, 30, 45, 60, 90, 120, 135, 180, 270, 360]
    for angle in angles:
        q = quaternion('Z', angle)
        q_str = quaternion_to_string(q)
        print(f"{angle:>3}°     quaternion('Z', {angle:>3})      {q_str}")


# Example usage:
if __name__ == "__main__":
    # Examples using MuJoCo quaternion creation
    print("MuJoCo Quaternion String Examples:")
    print("=" * 45)

    # Test different rotations
    test_cases = [
        ('Z', 0),
        ('Z', 45),
        ('Z', 90),
        ('Z', 180),
        ('X', 30),
        ('Y', 60)
    ]

    for axis, angle in test_cases:
        q = quaternion(axis, angle)
        q_str = quaternion_to_string(q)
        print(f"quaternion('{axis}', {angle:>3}°) = {q_str}")

    print("\n")

    # Print the MuJoCo quaternion lookup table
    print_mujoco_quaternion_table()

    # Show visualizations
    print("\nShowing visualizations...")

    # Single example
    fig1, ax1 = visualize_quaternion_2d('Z', 90)
    plt.show()

    # Grid of common rotations
    fig2 = demo_common_rotations()
    plt.show()

    # Quick string conversions for MuJoCo models
    print("\nMuJoCo Model Quaternions (Copy-Paste Ready):")
    print("-" * 45)
    common_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    for angle in common_angles:
        q_str = quaternion_to_string(quaternion('Z', angle))
        print(f"{angle:>3}°: {q_str}")