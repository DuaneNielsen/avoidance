import numpy as np
import mujoco


# Include the describe function from your library
def describe_quaternion(q, tolerance=1e-3):
    """
    Describe a quaternion in human-readable terms

    Parameters:
    q: numpy array [w, x, y, z] - quaternion
    tolerance: float - tolerance for considering values as zero/one

    Returns:
    str: human-readable description
    """
    w, x, y, z = q

    # Check if it's the identity quaternion
    if abs(w - 1) < tolerance and abs(x) < tolerance and abs(y) < tolerance and abs(z) < tolerance:
        return "No rotation (identity)"

    # Calculate rotation angle and axis
    angle = 2 * np.arccos(abs(w))
    angle_deg = np.degrees(angle)

    # Handle near-zero angle
    if angle < tolerance:
        return "No rotation (identity)"

    # Calculate rotation axis (normalize the vector part)
    vector_magnitude = np.sqrt(x * x + y * y + z * z)
    if vector_magnitude < tolerance:
        return "No rotation (identity)"

    axis_x = x / vector_magnitude
    axis_y = y / vector_magnitude
    axis_z = z / vector_magnitude

    # Determine primary axis
    primary_axis = ""
    if abs(axis_x) > 0.9:
        primary_axis = "X-axis"
    elif abs(axis_y) > 0.9:
        primary_axis = "Y-axis"
    elif abs(axis_z) > 0.9:
        primary_axis = "Z-axis"
    else:
        # Complex axis - describe the dominant components
        components = []
        if abs(axis_x) > tolerance:
            components.append(f"{axis_x:.2f}X")
        if abs(axis_y) > tolerance:
            components.append(f"{axis_y:.2f}Y")
        if abs(axis_z) > tolerance:
            components.append(f"{axis_z:.2f}Z")
        primary_axis = f"axis({'+'.join(components)})"

    return f"{angle_deg:.1f}° around {primary_axis}"


def common_rotation_names(q, tolerance=1e-2):
    """
    Check if quaternion matches common named rotations

    Returns:
    str or None: name if it matches a common rotation
    """
    common_rotations = {
        # Identity
        (1, 0, 0, 0): "Identity (no rotation)",

        # 90° rotations around cardinal axes
        (0.707, 0.707, 0, 0): "90° pitch up (around X)",
        (0.707, -0.707, 0, 0): "90° pitch down (around X)",
        (0.707, 0, 0.707, 0): "90° roll right (around Y)",
        (0.707, 0, -0.707, 0): "90° roll left (around Y)",
        (0.707, 0, 0, 0.707): "90° yaw right (around Z)",
        (0.707, 0, 0, -0.707): "90° yaw left (around Z)",

        # 180° rotations
        (0, 1, 0, 0): "180° pitch (around X)",
        (0, 0, 1, 0): "180° roll (around Y)",
        (0, 0, 0, 1): "180° yaw (around Z)",

        # 45° rotations
        (0.924, 0.383, 0, 0): "45° pitch up (around X)",
        (0.924, -0.383, 0, 0): "45° pitch down (around X)",
        (0.924, 0, 0.383, 0): "45° roll right (around Y)",
        (0.924, 0, -0.383, 0): "45° roll left (around Y)",
        (0.924, 0, 0, 0.383): "45° yaw right (around Z)",
        (0.924, 0, 0, -0.383): "45° yaw left (around Z)",

        # Common compound rotations
        (0.5, -0.5, 0.5, 0.5): "Sensor mount: 90° yaw + 45° pitch",
        (0.5, 0.5, 0.5, 0.5): "Camera orientation: looking down-right",
        (0.5, 0.5, -0.5, 0.5): "Camera orientation: looking up-right",
    }

    for reference, name in common_rotations.items():
        if all(abs(q[i] - reference[i]) < tolerance for i in range(4)):
            return name

    return None


def explain_quaternion(q):
    """
    Provide a comprehensive human-readable explanation of a quaternion

    Parameters:
    q: numpy array [w, x, y, z] - quaternion

    Returns:
    str: detailed explanation
    """
    # Check for common named rotations first
    common_name = common_rotation_names(q)
    if common_name:
        description = f"🎯 {common_name}"
    else:
        description = f"📐 {describe_quaternion(q)}"

    # Add coordinate frame transformation description
    try:
        # Convert to rotation matrix to see what happens to coordinate axes
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q.reshape(-1))
        R = R.reshape(3, 3)

        # See where the original X, Y, Z axes end up
        x_becomes = R @ np.array([1, 0, 0])
        y_becomes = R @ np.array([0, 1, 0])
        z_becomes = R @ np.array([0, 0, 1])

        def axis_direction(vec):
            tolerance = 0.9
            if abs(vec[0]) > tolerance:
                return "+X" if vec[0] > 0 else "-X"
            elif abs(vec[1]) > tolerance:
                return "+Y" if vec[1] > 0 else "-Y"
            elif abs(vec[2]) > tolerance:
                return "+Z" if vec[2] > 0 else "-Z"
            else:
                return f"({vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f})"

        frame_info = (
            f"\n📦 Frame transformation:\n"
            f"   X-axis → {axis_direction(x_becomes)}\n"
            f"   Y-axis → {axis_direction(y_becomes)}\n"
            f"   Z-axis → {axis_direction(z_becomes)}"
        )

        description += frame_info

    except:
        pass  # Skip frame transformation if MuJoCo not available

    return description


# Quaternion library functions
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
    """Roll rotation (around X-axis)"""
    return rot_x(degrees)


def yaw(degrees):
    """Yaw rotation (around Z-axis)"""
    return rot_z(degrees)


def pitch(degrees):
    """Pitch rotation (around Y-axis)"""
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


def demo_quaternion(name, q):
    """Demo a single quaternion with nice formatting"""
    print(f"\n🔄 {name}")
    print(f"   Quaternion: [{quat_string(q)}]")
    print(f"   {explain_quaternion(q)}")


def main():
    """Comprehensive quaternion demo"""
    print("=" * 80)
    print("🚀 QUATERNION LIBRARY DEMO")
    print("=" * 80)

    # 1. Basic rotations
    print("\n📍 BASIC ROTATIONS")
    print("-" * 40)
    demo_quaternion("Identity (no rotation)", np.array([1, 0, 0, 0]))
    demo_quaternion("45° around X-axis", rot_x(45))
    demo_quaternion("90° around Y-axis", rot_y(90))
    demo_quaternion("180° around Z-axis", rot_z(180))
    demo_quaternion("30° around arbitrary axis [1,1,1]", create_rotation_quat([1, 1, 1], 30))

    # 2. Named rotations (roll, pitch, yaw)
    print("\n✈️  ROLL, PITCH, YAW")
    print("-" * 40)
    demo_quaternion("Roll 45°", roll(45))
    demo_quaternion("Pitch 30°", pitch(30))
    demo_quaternion("Yaw 90°", yaw(90))

    # 3. Axis flipping
    print("\n🔄 AXIS FLIPPING")
    print("-" * 40)
    demo_quaternion("Flip X-axis", flip_x())
    demo_quaternion("Flip Y-axis", flip_y())
    demo_quaternion("Flip Z-axis", flip_z())

    # 4. Axis alignment
    print("\n🎯 AXIS ALIGNMENT")
    print("-" * 40)
    demo_quaternion("Align X to Y", align_x_to_y())
    demo_quaternion("Align X to Z", align_x_to_z())
    demo_quaternion("Align Z to Y", align_z_to_y())

    # 5. Quaternion composition
    print("\n🔗 QUATERNION COMPOSITION")
    print("-" * 40)

    # Example: Camera looking down and slightly rotated
    camera_down = compose(rot_x(45), rot_z(30))
    demo_quaternion("Camera: 45° pitch down + 30° yaw", camera_down)

    # Example: Sensor mount
    sensor_mount = compose(rot_z(90), rot_x(45))
    demo_quaternion("Sensor: 90° yaw + 45° pitch", sensor_mount)

    # Example: Complex transformation
    complex_transform = compose(rot_x(30), rot_y(45), rot_z(60))
    demo_quaternion("Complex: 30°X + 45°Y + 60°Z", complex_transform)

    # 6. Real-world examples
    print("\n🌍 REAL-WORLD EXAMPLES")
    print("-" * 40)

    # Vehicle sensor array (common in robotics)
    forward_sensor = rot_z(0)  # Identity - pointing forward
    left_sensor = rot_z(45)  # 45° to the left
    right_sensor = rot_z(-45)  # 45° to the right

    demo_quaternion("Vehicle forward sensor", forward_sensor)
    demo_quaternion("Vehicle left sensor (+45°)", left_sensor)
    demo_quaternion("Vehicle right sensor (-45°)", right_sensor)

    # Drone camera gimbal
    gimbal_level = rot_x(0)  # Level
    gimbal_down = rot_x(90)  # Looking straight down
    gimbal_angled = rot_x(30)  # 30° down

    demo_quaternion("Drone camera level", gimbal_level)
    demo_quaternion("Drone camera straight down", gimbal_down)
    demo_quaternion("Drone camera 30° down", gimbal_angled)

    # 7. Advanced compositions
    print("\n🚁 ADVANCED COMPOSITIONS")
    print("-" * 40)

    # Simulate a drone that's tilted and its camera is gimbal-stabilized
    drone_attitude = compose(rot_x(15), rot_y(10))  # Drone is tilted
    camera_compensation = compose(rot_x(-15), rot_y(-10))  # Camera compensates
    stabilized_camera = compose(drone_attitude, camera_compensation)

    demo_quaternion("Tilted drone attitude", drone_attitude)
    demo_quaternion("Camera compensation", camera_compensation)
    demo_quaternion("Net camera orientation", stabilized_camera)

    # 8. Testing quaternion properties
    print("\n🔬 QUATERNION PROPERTIES")
    print("-" * 40)

    # Test quaternion magnitude (should be 1.0 for valid rotations)
    test_quat = compose(rot_x(45), rot_y(30), rot_z(60))
    magnitude = np.linalg.norm(test_quat)
    print(f"\n📏 Quaternion magnitude test:")
    print(f"   Test quaternion: [{quat_string(test_quat)}]")
    print(f"   Magnitude: {magnitude:.6f} (should be ≈1.0)")
    print(f"   ✓ Valid" if abs(magnitude - 1.0) < 1e-6 else "   ✗ Invalid")

    # Test inverse property: q * q^(-1) = identity
    q = rot_z(45)
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])  # Quaternion inverse
    identity_test = quaternion_multiply(q, q_inv)
    print(f"\n🔄 Inverse property test:")
    print(f"   q = [{quat_string(q)}]")
    print(f"   q^(-1) = [{quat_string(q_inv)}]")
    print(f"   q * q^(-1) = [{quat_string(identity_test)}]")
    print(f"   ✓ Identity achieved" if abs(identity_test[0] - 1.0) < 1e-6 else "   ✗ Not identity")

    print("\n" + "=" * 80)
    print("🎉 DEMO COMPLETE!")
    print("=" * 80)
    print("\n💡 Key takeaways:")
    print("   • Quaternions represent rotations as [w, x, y, z]")
    print("   • w = cos(θ/2), [x,y,z] = sin(θ/2) * rotation_axis")
    print("   • Composition: multiply quaternions to combine rotations")
    print("   • Always normalize quaternions to maintain unit magnitude")
    print("   • Use the describe functions to understand complex rotations!")


if __name__ == "__main__":
    main()