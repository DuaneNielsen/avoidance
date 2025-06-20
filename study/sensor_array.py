import mujoco
import mujoco.viewer
import numpy as np
import time

"""
Complete demo: 2D vehicle with heightfield terrain and ghost vector visualization
- Vehicle with forward/reverse/brake and rotational steering
- Heightmap terrain with hills and flat spawn area
- Fan of rangefinder sensors
- Ghost vector visualization using scene graph (no physics interference)
"""

SENSOR_ANGLE_DEGREES = 64
NUM_SENSORS = 64
GOAL_POS = "10. 0. 0.2"
VEHICLE_START_POS = "0 0 0.2"


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


sensor_site_xml = ""
sensor_rangefinders_xml = ""
rangefinder_angles = np.linspace(start=-SENSOR_ANGLE_DEGREES, stop=SENSOR_ANGLE_DEGREES, num=NUM_SENSORS)

for i, theta in enumerate(rangefinder_angles):
    sensor_site_xml += \
        f"""
        <site name="site_rangefinder{i}" quat="{quat_string(roll(theta))}" size="0.01" rgba="1 0 0 1"/>
        """
    sensor_rangefinders_xml += \
        f"""
        <rangefinder name="rangefinder{i}" site="site_rangefinder{i}"/>
        """

xml = f"""
<mujoco model="vehicle_with_terrain">
  <option timestep="0.01"/>

  <asset>
    <!-- Heightfield terrain -->
    <hfield name="terrain" nrow="128" ncol="128" size="10 10 2.0 0.1"/>
    <texture name="terrain_texture" type="2d" builtin="checker" 
             width="512" height="512" rgb2="0.2 0.4 0.2" rgb1="0.6 0.8 0.6"/>
    <material name="terrain_material" texture="terrain_texture" texrepeat="16 16" 
              texuniform="true" reflectance="0.3"/>

    <material name="vehicle_material" rgba="0.2 0.8 0.2 1"/>
    <material name="goal_material" rgba="0.3 1.0 0.3 1"/>
  </asset>

  <worldbody>
    <light name="top" pos="0 0 8" dir="0 0 -1"/>

    <!-- Heightfield terrain -->
    <geom name="terrain" type="hfield" hfield="terrain" material="terrain_material"
          friction="0.8 0.1 0.1"/>

    <body name="vehicle" pos="{VEHICLE_START_POS}">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="1.0"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="1.0"/>
      <joint name="rotate_z" type="hinge" axis="0 0 1" damping="0.5"/>

      <geom name="box" type="box" size="0.3 0.15 0.05" material="vehicle_material"
            friction="0.8 0.1 0.1"/>

      <site name="control_site" pos="0 0 0" size="0.02" rgba="1 0 0 1" />

      <frame pos="0.3 0. 0" quat="{quat_string(pitch(90))}">
         {sensor_site_xml}
      </frame>
    </body>

    <!-- Goal sphere -->
    <geom name="goal" pos="{GOAL_POS}" type="sphere" size="0.07" material="goal_material" />

  </worldbody>

  <actuator>
    <!-- Drive actuators -->
    <velocity name="drive" site="control_site" kv="8.0" gear="1 0 0 0 0 0" 
              ctrlrange="-2.0 2.0"/>
    <velocity name="steer" joint="rotate_z" kv="5.0" ctrlrange="-3.0 3.0"/>

    <!-- Brake actuators - dampers that resist motion -->
    <damper name="brake_x" joint="slide_x" kv="40.0" ctrlrange="0 1"/>
    <damper name="brake_y" joint="slide_y" kv="40.0" ctrlrange="0 1"/>
    <damper name="brake_rot" joint="rotate_z" kv="32.0" ctrlrange="0 1"/>
  </actuator>

  <sensor>
    <framepos name="vehicle_pos" objtype="body" objname="vehicle"/>
    <framepos name="goal_pos" objtype="geom" objname="goal"/>
    <framepos name="goalvec" objtype="geom" objname="goal" reftype="site" refname="control_site"/>
    {sensor_rangefinders_xml}
  </sensor>

</mujoco>
"""


def generate_terrain_with_flat_center(nrow, ncol, hills_x=6, hills_y=6,
                                      hill_height=0.6, hill_radius=0.25,
                                      flat_radius=1.5):
    """Generate heightfield terrain with hills but flatten the center region for vehicle spawn."""
    terrain = np.zeros((nrow, ncol))

    # World size from XML: size="10 10 2.0 0.1" means 20x20 world units
    world_size_x = 20.0
    world_size_y = 20.0

    # Calculate hill positions
    spacing_x = nrow / (hills_x + 1)
    spacing_y = ncol / (hills_y + 1)

    # Center of the grid
    center_x = nrow / 2
    center_y = ncol / 2

    # Convert flat radius from world units to grid units
    flat_radius_grid_x = flat_radius * nrow / world_size_x
    flat_radius_grid_y = flat_radius * ncol / world_size_y

    # Generate hills
    for i in range(hills_x):
        for j in range(hills_y):
            hill_center_x = (i + 1) * spacing_x
            hill_center_y = (j + 1) * spacing_y

            # Skip hills too close to center
            dist_from_center = np.sqrt((hill_center_x - center_x) ** 2 + (hill_center_y - center_y) ** 2)
            if dist_from_center < max(flat_radius_grid_x, flat_radius_grid_y) * 1.5:
                continue

            # Create circular hill
            for x in range(nrow):
                for y in range(ncol):
                    dist = np.sqrt((x - hill_center_x) ** 2 + (y - hill_center_y) ** 2)
                    radius = hill_radius * min(spacing_x, spacing_y)

                    if dist < radius:
                        height = hill_height * (np.cos(np.pi * dist / radius) + 1) / 2
                        terrain[x, y] = max(terrain[x, y], height)

    # Flatten the center region for vehicle spawn
    for x in range(nrow):
        for y in range(ncol):
            dist_x = abs(x - center_x)
            dist_y = abs(y - center_y)

            if dist_x < flat_radius_grid_x and dist_y < flat_radius_grid_y:
                terrain[x, y] = 0.0
            elif dist_x < flat_radius_grid_x * 1.2 and dist_y < flat_radius_grid_y * 1.2:
                fade_factor = max(
                    (dist_x - flat_radius_grid_x) / (flat_radius_grid_x * 0.2),
                    (dist_y - flat_radius_grid_y) / (flat_radius_grid_y * 0.2)
                )
                fade_factor = np.clip(fade_factor, 0.0, 1.0)
                terrain[x, y] *= fade_factor

    print(f"Generated {hills_x}x{hills_y} hills with flat center")
    print(f"Terrain height range: {terrain.min():.3f} to {terrain.max():.3f}")
    print(f"Flat radius: {flat_radius:.1f} world units")

    return terrain.flatten()


def add_vector_to_scene(scene, start_pos, end_pos, rgba=(1, 0, 0, 1), width=0.005):
    """Add a vector line directly to the scene graph (no physics collision)"""
    if scene.ngeom >= scene.maxgeom:
        return

    # Initialize new visual geometry
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),  # size (will be set by connector)
        np.zeros(3),  # pos (will be set by connector)
        np.zeros(9),  # mat (will be set by connector)
        np.array(rgba, dtype=np.float32)
    )

    # Use mjv_connector to create line between points
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        start_pos,
        end_pos
    )

    scene.ngeom += 1


def add_marker_to_scene(scene, pos, rgba=(1, 1, 0, 1), size=0.05):
    """Add a spherical marker to the scene graph"""
    if scene.ngeom >= scene.maxgeom:
        return

    # Create sphere marker
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, 0, 0]),  # size for sphere
        pos,  # position
        np.eye(3).flatten(),  # identity rotation matrix
        np.array(rgba, dtype=np.float32)
    )

    scene.ngeom += 1


def site_local_to_world_simple(model, data, site_name, local_coords):
    """Transform site-local coordinates to world coordinates."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    body_id = model.site_bodyid[site_id]

    world_pos = np.zeros(3)
    world_mat = np.zeros(9)

    # Identity quaternion since we just want position transformation
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

    mujoco.mj_local2Global(data, world_pos, world_mat, local_coords, identity_quat, body_id, 0)

    return world_pos


# Create model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Generate terrain with flattened center
nrow, ncol = 128, 128
terrain_data = generate_terrain_with_flat_center(
    nrow=nrow,
    ncol=ncol,
    hills_x=6,
    hills_y=6,
    hill_height=0.6,
    hill_radius=0.25,
    flat_radius=1.5
)

# Set heightfield data
model.hfield_data[:] = terrain_data

timestep = 1 / 60.0

# Control inputs
forward_vel = 0.0
rotate_vel = 0.0
brake_force = 0.0


def key_callback(keycode):
    global forward_vel, rotate_vel, brake_force

    if keycode == 32:  # Spacebar - BRAKE
        forward_vel = 0.0
        rotate_vel = 0.0
        brake_force = 1.0
        print("BRAKING")
    elif keycode == 265:  # Up arrow - forward
        forward_vel = 1.0
        rotate_vel = 0.0
        brake_force = 0.0
        print("FORWARD")
    elif keycode == 264:  # Down arrow - backward
        forward_vel = -1.0
        rotate_vel = 0.0
        brake_force = 0.0
        print("BACKWARD")
    elif keycode == 263:  # Left arrow - rotate left
        rotate_vel = 1.0
        brake_force = 0.0
        print("ROTATE LEFT")
    elif keycode == 262:  # Right arrow - rotate right
        rotate_vel = -1.0
        brake_force = 0.0
        print("ROTATE RIGHT")


print("Vehicle with Ghost Vector Visualization Demo")
print("=" * 50)
print("Controls:")
print("  ↑ : Forward")
print("  ↓ : Backward")
print("  ← : Rotate Left")
print("  → : Rotate Right")
print("  SPACE : BRAKE (stops all movement)")
print()
print("Features:")
print("- Blue vector: Vehicle to goal (ghost object, no physics)")
print("- Yellow marker: Vector midpoint")
print("- Red rays: Rangefinder sensors")
print("- Ghost objects don't interfere with sensors or physics!")
print("=" * 50)

# Run simulation with ghost vector visualization
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    step_count = 0
    last_sensor_print = 0

    while viewer.is_running():
        start = time.time()

        # Set visualization flags
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True

        # Apply controls
        data.ctrl[0] = forward_vel  # Drive
        data.ctrl[1] = rotate_vel  # Steer
        data.ctrl[2] = brake_force  # X brake
        data.ctrl[3] = brake_force  # Y brake
        data.ctrl[4] = brake_force  # Rotation brake

        # Step physics
        mujoco.mj_step(model, data)

        # Get vehicle and goal positions from sensors
        # vehicle_pos = data.sensordata[0:3]
        # goal_pos = data.sensordata[3:6]
        vector_to_goal_vehicle_frame = data.sensordata[6:9]

        goal_pos = site_local_to_world_simple(model, data, 'control_site', vector_to_goal_vehicle_frame)
        vehicle_pos = site_local_to_world_simple(model, data, 'control_site', np.zeros(3))

        # Calculate vector properties
        vector = goal_pos - vehicle_pos
        distance = np.linalg.norm(vector)
        midpoint = (vehicle_pos + goal_pos) / 2

        midpoint = (vehicle_pos + goal_pos) / 2

        # Reset scene to original model geometry only
        viewer.user_scn.ngeom = model.ngeom

        # Add ghost visualization (pure visual, no physics interference)

        # 1. Main vector from vehicle to goal
        add_vector_to_scene(
            viewer.user_scn,
            vehicle_pos,
            goal_pos,
            rgba=(0, 0, 1, 0.8),  # Blue, semi-transparent
            width=0.02
        )

        # 2. Midpoint marker
        add_marker_to_scene(
            viewer.user_scn,
            midpoint,
            rgba=(1, 1, 0, 1),  # Yellow
            size=0.08
        )

        # 3. Additional visualization: distance indicators
        if distance > 5.0:
            # Add intermediate markers for long distances
            quarter_point = vehicle_pos + vector * 0.25
            three_quarter_point = vehicle_pos + vector * 0.75

            add_marker_to_scene(viewer.user_scn, quarter_point,
                                rgba=(0, 1, 1, 0.6), size=0.05)
            add_marker_to_scene(viewer.user_scn, three_quarter_point,
                                rgba=(1, 0, 1, 0.6), size=0.05)

        # 4. Progress indicator: vertical line at vehicle showing progress
        progress_height = min(distance / 10.0, 1.0)  # Scale to 0-1
        progress_top = vehicle_pos + np.array([0, 0, progress_height])
        add_vector_to_scene(
            viewer.user_scn,
            vehicle_pos + np.array([0, 0, 0.1]),
            progress_top,
            rgba=(0, 1, 0, 0.9),  # Green progress bar
            width=0.03
        )

        # Print status every 60 steps
        if step_count - last_sensor_print >= 60:
            print(f"Distance to goal: {distance:.2f}m | Vehicle pos: ({vehicle_pos[0]:.1f}, {vehicle_pos[1]:.1f}) | "
                  f"Forward sensor: {data.sensordata[6]:.2f}m")
            last_sensor_print = step_count

        step_count += 1
        viewer.sync()

        # Frame rate control
        curr = time.time()
        while curr - start < timestep:
            time.sleep(0.001)
            curr = time.time()

print("\nDemo completed!")
print("Key takeaways:")
print("- Ghost vectors don't block rangefinder rays")
print("- Scene graph visualization has zero physics overhead")
print("- Perfect for dynamic navigation displays")
