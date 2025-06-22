import mujoco
import mujoco.viewer
import numpy as np
import time
import quaterion as quat

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


VEHICLE_LENGTH = 0.3
VEHICLE_WIDTH = 0.1
VEHICLE_HEIGHT = 0.05
VEHICLE_COLLISION = 0.1
VEHICLE_CLEARANCE = 0.01
VEHICLE_COLLISION_HEIGHT = - VEHICLE_HEIGHT + VEHICLE_CLEARANCE
VEHICLE_SIZE = f"{VEHICLE_LENGTH} {VEHICLE_WIDTH} {VEHICLE_HEIGHT - VEHICLE_CLEARANCE}"

VEHICLE_START_POS = f"0 0 {VEHICLE_HEIGHT}"

VEHICLE_COLLISION_LEFT_SIDE  = f"{- VEHICLE_LENGTH - VEHICLE_COLLISION} {VEHICLE_WIDTH + VEHICLE_COLLISION} {VEHICLE_COLLISION_HEIGHT}"
VEHICLE_COLLISION_RIGHT_SIDE  = f"{- VEHICLE_LENGTH - VEHICLE_COLLISION} {-VEHICLE_WIDTH - VEHICLE_COLLISION} {VEHICLE_COLLISION_HEIGHT}"

VEHICLE_COLLISION_FRONT  = f"{VEHICLE_LENGTH + VEHICLE_COLLISION} {-VEHICLE_WIDTH - VEHICLE_COLLISION} {VEHICLE_COLLISION_HEIGHT}"
VEHICLE_COLLISION_BACK  = f"{- VEHICLE_LENGTH - VEHICLE_COLLISION} {-VEHICLE_WIDTH - VEHICLE_COLLISION} {VEHICLE_COLLISION_HEIGHT}"



sensor_site_xml = ""
sensor_rangefinders_xml = ""
rangefinder_angles = np.linspace(start=-SENSOR_ANGLE_DEGREES, stop=SENSOR_ANGLE_DEGREES, num=NUM_SENSORS)

for i, theta in enumerate(rangefinder_angles):
    sensor_site_xml += \
        f"""
        <site name="site_rangefinder{i}" quat="{quat.quat_string(quat.roll(theta))}" size="0.01" rgba="1 0 0 1"/>
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

      <geom name="vehicle_body" type="box" size="{VEHICLE_SIZE}" material="vehicle_material"
            friction="0.8 0.1 0.1"/>

      <site name="control_site" pos="0 0 0" size="0.02" rgba="1 0 0 1" />

      <frame pos="0.3 0. 0" quat="{quat.quat_string(quat.pitch(90))}">
         {sensor_site_xml}         
      </frame>
      
      <site name="sensor_fl" pos="{VEHICLE_COLLISION_LEFT_SIDE}" size="0.01" rgba="1 0 0 1" 
            quat="{quat.quat_string(quat.pitch(90))}"/>
            
      <site name="sensor_fr" pos="{VEHICLE_COLLISION_RIGHT_SIDE}" size="0.01" rgba="1 0 0 1" 
            quat="{quat.quat_string(quat.pitch(90))}"/>

      <site name="sensor_rl" pos="{VEHICLE_COLLISION_FRONT}" size="0.01" rgba="0 0 1 1" 
            quat="{quat.quat_string(quat.compose(quat.pitch(90), quat.roll(-90)))}"/>

      <site name="sensor_rr" pos="{VEHICLE_COLLISION_BACK}" size="0.01" rgba="0 0 1 1" 
            quat="{quat.quat_string(quat.compose(quat.pitch(90), quat.roll(-90)))}"/>
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
    
    <!-- Corner rangefinder sensors -->
    <rangefinder name="range_fl" site="sensor_fl" cutoff="{(VEHICLE_LENGTH + VEHICLE_COLLISION) * 2}"/>
    <rangefinder name="range_fr" site="sensor_fr" cutoff="{(VEHICLE_LENGTH + VEHICLE_COLLISION) * 2}"/>
    <rangefinder name="range_rl" site="sensor_rl" cutoff="{(VEHICLE_WIDTH + VEHICLE_COLLISION) * 2}"/>
    <rangefinder name="range_rr" site="sensor_rr" cutoff="{(VEHICLE_WIDTH + VEHICLE_COLLISION) * 2}"/>
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


def read_collision_sensors(data):
    return np.abs(data.sensordata[9:11]), np.abs(data.sensordata[11:13])


def collision_detected(data):
    left_right, front_rear = read_collision_sensors(data)
    side_collision = left_right < (VEHICLE_LENGTH + VEHICLE_COLLISION) * 2
    front_rear_collision = front_rear < (VEHICLE_WIDTH + VEHICLE_COLLISION) * 2
    return np.any(side_collision | front_rear_collision)

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
        vehicle_collision = collision_detected(data)
        vehicle_body_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "vehicle_body")
        red, green = vehicle_collision, ~vehicle_collision
        model.geom_rgba[vehicle_body_geom_id] = [red, green, 0., 1.0]


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
