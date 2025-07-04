import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
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

VEHICLE_COLLISION_LEFT_SENSOR_NAME = 'range_left'
VEHICLE_COLLISION_RIGHT_SENSOR_NAME = 'range_right'
VEHICLE_COLLISION_FRONT_SENSOR_NAME = 'range_front'
VEHICLE_COLLISION_BACK_SENSOR_NAME = 'range_back'
GOAL_SENSOR_NAME = 'goalvec'
RANGEFINDER_SENSOR_PREFIX = 'rangefinder'
RANGEFINDER_SITE_PREFIX = 'site_rangefinder'
RANGEFINDER_CUTOFF = 4.83

CAMERA_TRACK_VEHICLE = 'track_vehicle'
CAMERA_PERSPECTIVE = 'perspective'
CAMERA_FIRST_PERSON = 'first_person'

sensor_site_xml = ""
sensor_rangefinders_xml = ""
rangefinder_angles = np.linspace(start=-SENSOR_ANGLE_DEGREES, stop=SENSOR_ANGLE_DEGREES, num=NUM_SENSORS)

for i, theta in enumerate(rangefinder_angles):
    sensor_site_xml += \
        f"""
        <site name="{RANGEFINDER_SITE_PREFIX}{i}" quat="{quat.quat_string(quat.roll(theta))}" size="0.01" rgba="1 0 0 1"/>
        """
    sensor_rangefinders_xml += \
        f"""
        <rangefinder name="{RANGEFINDER_SENSOR_PREFIX}{i}" site="site_rangefinder{i}" cutoff="{RANGEFINDER_CUTOFF}"/>
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
    <camera name="{CAMERA_PERSPECTIVE}" pos="-7.028 -5.392 6.348" xyaxes="0.609 -0.793 0.000 0.462 0.355 0.813"/>
    

    <!-- Heightfield terrain -->
    <geom name="terrain" type="hfield" hfield="terrain" material="terrain_material"
          friction="0.8 0.1 0.1"/>

    <body name="vehicle" pos="{VEHICLE_START_POS}">
      <camera name="{CAMERA_TRACK_VEHICLE}" pos="0 0 10" mode="track"/>
      <camera name="{CAMERA_FIRST_PERSON}" pos="-1 0 0.3" xyaxes="0 -1 0 0 0.3 1"/>
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
    <framepos name="{GOAL_SENSOR_NAME}" objtype="geom" objname="goal" reftype="site" refname="control_site"/>
    
    <!-- Corner rangefinder sensors -->
    <rangefinder name="{VEHICLE_COLLISION_LEFT_SENSOR_NAME}" site="sensor_fl" cutoff="{(VEHICLE_LENGTH + VEHICLE_COLLISION) * 2}"/>
    <rangefinder name="{VEHICLE_COLLISION_RIGHT_SENSOR_NAME}" site="sensor_fr" cutoff="{(VEHICLE_LENGTH + VEHICLE_COLLISION) * 2}"/>
    <rangefinder name="{VEHICLE_COLLISION_FRONT_SENSOR_NAME}" site="sensor_rl" cutoff="{(VEHICLE_WIDTH + VEHICLE_COLLISION) * 2}"/>
    <rangefinder name="{VEHICLE_COLLISION_BACK_SENSOR_NAME}" site="sensor_rr" cutoff="{(VEHICLE_WIDTH + VEHICLE_COLLISION) * 2}"/>
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


def get_sensor_data_range(model, data, sensor_name):
    """Get the full data range for a multi-dimensional sensor"""
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    assert sensor_id != -1, f"sensor {sensor_name} not found"
    start_idx = model.sensor_adr[sensor_id]  # Use sensor_adr, not sum of dims!
    end_idx = start_idx + model.sensor_dim[sensor_id]
    return data.sensordata[start_idx:end_idx]


def read_collision_sensors(model, data):
    left = get_sensor_data_range(model, data, VEHICLE_COLLISION_LEFT_SENSOR_NAME)
    right = get_sensor_data_range(model, data, VEHICLE_COLLISION_RIGHT_SENSOR_NAME)
    front = get_sensor_data_range(model, data, VEHICLE_COLLISION_FRONT_SENSOR_NAME)
    back = get_sensor_data_range(model, data, VEHICLE_COLLISION_BACK_SENSOR_NAME)
    return np.abs(left), np.abs(right), np.abs(front), np.abs(back)


def read_goal_sensor(model, data):
    goal_xyz_in_vehicle_frame = get_sensor_data_range(model, data, GOAL_SENSOR_NAME)
    goal_xy_in_vehicle_frame = goal_xyz_in_vehicle_frame[0:2]
    distance = np.linalg.norm(goal_xy_in_vehicle_frame)
    normalized_goal_xy_in_vehicle_frame = goal_xy_in_vehicle_frame / distance
    angle = np.arctan(normalized_goal_xy_in_vehicle_frame[1:2], normalized_goal_xy_in_vehicle_frame[0:1])
    return distance, angle


def goal_angle_to_onehot(model, data, num_bins=32):
    """Convert goal angle to one-hot encoding with 32 bins.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        num_bins: Number of angular bins (default 32)
    
    Returns:
        numpy array: One-hot encoding where:
        - bin 0: goal is on the left side (-90°)
        - bins 15-16: goal is dead ahead (0°) 
        - bin 31: goal is on the right side (+90°)
    """
    distance, angle = read_goal_sensor(model, data)
    
    # Convert angle from [-pi/2, pi/2] to [0, num_bins-1]
    # angle is in radians, where -pi/2 is left, 0 is ahead, +pi/2 is right
    angle_degrees = np.degrees(angle[0])  # Convert to degrees and extract scalar
    
    # Map from [-90, +90] degrees to [31, 0] (reversed)
    # -90° -> bin 31, 0° -> bin 15.5 (rounds to 15 or 16), +90° -> bin 0
    normalized_angle = (angle_degrees + 90.0) / 180.0  # Map [-90,90] to [0,1]
    bin_index = int((1.0 - normalized_angle) * (num_bins - 1))  # Reverse and map [0,1] to [31,0]
    
    # Clamp to valid range
    bin_index = np.clip(bin_index, 0, num_bins - 1)
    
    # Create one-hot encoding
    onehot = np.zeros(num_bins)
    onehot[bin_index] = 1.0
    
    return onehot


def read_rangefinder_array(model_cpu, data):
    sensor_id = mujoco.mj_name2id(model_cpu, mujoco.mjtObj.mjOBJ_SENSOR, RANGEFINDER_SENSOR_PREFIX + '0')
    RANGEFINDER_0 = model_cpu.sensor_adr[sensor_id]
    return data.sensordata[RANGEFINDER_0:]


def normalize_rangefinders(rangefinder_array):
    rangefinder_norm = np.where(rangefinder_array == -1., RANGEFINDER_CUTOFF, rangefinder_array) / RANGEFINDER_CUTOFF
    return 1. - rangefinder_norm


def create_rangefinder_visualization(normalized_ranges, goal_onehot=None, strip_height=80, strip_width=None):
    """Create a horizontal strip visualization of rangefinder data with optional goal indicator.
    
    Args:
        normalized_ranges: Array of normalized rangefinder values (0-1)
        goal_onehot: Optional one-hot encoded goal direction (32 bins)
        strip_height: Height of the visualization strip in pixels
        strip_width: Width per sensor (if None, auto-calculated)
    
    Returns:
        BGR image array for OpenCV display
    """
    num_sensors = len(normalized_ranges)
    
    if strip_width is None:
        strip_width = max(16, 1024 // num_sensors)  # At least 16 pixels per sensor, max 1024 total width
    
    total_width = num_sensors * strip_width
    
    # Calculate total height: rangefinder strip + goal strip
    goal_strip_height = 40 if goal_onehot is not None else 0
    total_height = strip_height + goal_strip_height
    
    # Create grayscale image: 0 = black (far), 255 = white (close)
    grayscale_values = (normalized_ranges * 255).astype(np.uint8)
    
    # Create the rangefinder strip by repeating each value strip_width times horizontally
    rangefinder_strip = np.repeat(grayscale_values.reshape(1, -1), strip_height, axis=0)
    rangefinder_strip = np.repeat(rangefinder_strip, strip_width, axis=1)
    
    # Convert to BGR for OpenCV (all channels same for grayscale)
    bgr_strip = cv2.cvtColor(rangefinder_strip, cv2.COLOR_GRAY2BGR)
    
    # Add goal visualization if provided
    if goal_onehot is not None:
        # Create goal strip with blue color where goal is detected
        goal_strip = np.zeros((goal_strip_height, total_width, 3), dtype=np.uint8)
        
        # Map 32 goal bins to sensor positions
        bins_per_sensor = len(goal_onehot) / num_sensors
        
        for i, goal_value in enumerate(goal_onehot):
            if goal_value > 0:  # Goal detected in this bin
                # Map goal bin to sensor range
                start_sensor = int(i / bins_per_sensor)
                end_sensor = int((i + 1) / bins_per_sensor)
                
                for sensor_idx in range(start_sensor, min(end_sensor + 1, num_sensors)):
                    x_start = sensor_idx * strip_width
                    x_end = (sensor_idx + 1) * strip_width
                    goal_strip[:, x_start:x_end] = [255, 0, 0]  # Blue in BGR format
        
        # Combine rangefinder and goal strips vertically
        bgr_strip = np.vstack([bgr_strip, goal_strip])
    
    # Add sensor index labels every 10th sensor
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    text_color = (0, 255, 0)  # Green text
    
    for i in range(0, num_sensors, 10):
        x_pos = i * strip_width + 2
        y_pos = strip_height - 5
        cv2.putText(bgr_strip, str(i), (x_pos, y_pos), font, font_scale, text_color, font_thickness)
    
    return bgr_strip


def collision_detected(model, data):
    left, right, front, rear = read_collision_sensors(model, data)
    left_collision = left < (VEHICLE_LENGTH + VEHICLE_COLLISION) * 2
    right_collision = right < (VEHICLE_LENGTH + VEHICLE_COLLISION) * 2
    front_collision = front < (VEHICLE_WIDTH + VEHICLE_COLLISION) * 2
    rear_collision = rear < (VEHICLE_WIDTH + VEHICLE_COLLISION) * 2
    return left_collision | right_collision | front_collision | rear_collision

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

# Method 1: Simple list of all sensor names
def list_sensor_names(model):
    """Get a list of all sensor names"""
    sensor_names = []
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        sensor_names.append(name)
    return sensor_names


if __name__ == '__main__':


    # Create model and data
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    sensor_names = list_sensor_names(model)
    print("All sensors:", sensor_names)


    goal_vec_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'goal_vec')


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
    auto_mode = False

    def key_callback(keycode):
        global forward_vel, rotate_vel, brake_force, auto_mode

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
        elif keycode == 344: # Right shift, toggle PID control
            auto_mode = not auto_mode

    print("Vehicle with Ghost Vector Visualization Demo")
    print("=" * 50)
    print("Controls:")
    print("  ↑ : Forward")
    print("  ↓ : Backward")
    print("  ← : Rotate Left")
    print("  → : Rotate Right")
    print("  SPACE : BRAKE (stops all movement)")
    print("  Right Shift : Toggle auto-mode")
    print()
    print("Features:")
    print("- Blue vector: Vehicle to goal (ghost object, no physics)")
    print("- Red rays: Rangefinder sensors in MuJoCo viewer")
    print("- Rangefinder CV2 window: White = close obstacles, Black = far/no obstacles")
    print("- Green numbers show sensor indices (0-63)")
    print("- Ghost objects don't interfere with sensors or physics!")
    print("=" * 50)

    from collections import deque
    rot_I = deque(maxlen=5)

    # Run simulation with ghost vector visualization
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Set default camera to first person view
        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_FIRST_PERSON)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
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

            if auto_mode:
                # dodgy control scheme
                distance, angle = read_goal_sensor(model, data)

                rot_P = 0.5 * angle
                steer_angle = rot_P + sum(rot_I)
                rot_I.appendleft(rot_P)
                data.ctrl[1] = steer_angle[0]
                data.ctrl[0] = np.maximum(0.3 * distance, 2.0)

            # Step physics
            mujoco.mj_step(model, data)

            rangefinder = read_rangefinder_array(model, data)
            rangefinder_normalized = normalize_rangefinders(rangefinder)
            
            # Get goal direction as one-hot encoding
            goal_onehot = goal_angle_to_onehot(model, data)
            
            # Create and display rangefinder visualization with goal indicator
            rangefinder_viz = create_rangefinder_visualization(rangefinder_normalized, goal_onehot)
            cv2.imshow('Rangefinder Sensors', rangefinder_viz)
            cv2.waitKey(1)  # Non-blocking update
            
            # Also save visualization every 30 steps for debugging
            if step_count % 30 == 0:
                cv2.imwrite(f'rangefinder_viz_step_{step_count}.png', rangefinder_viz)
                print(f"Saved rangefinder visualization at step {step_count}")

            # Get vehicle and goal positions from sensors
            # vehicle_pos = data.sensordata[0:3]
            # goal_pos = data.sensordata[3:6]
            vector_to_goal_vehicle_frame = data.sensordata[6:9]

            goal_pos = site_local_to_world_simple(model, data, 'control_site', vector_to_goal_vehicle_frame)
            vehicle_pos = site_local_to_world_simple(model, data, 'control_site', np.zeros(3))
            vehicle_collision = collision_detected(model, data)
            vehicle_body_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "vehicle_body")
            red, green = vehicle_collision, ~vehicle_collision
            model.geom_rgba[vehicle_body_geom_id] = [red[0], green[0], 0., 1.0]


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
            # add_marker_to_scene(
            #     viewer.user_scn,
            #     midpoint,
            #     rgba=(1, 1, 0, 1),  # Yellow
            #     size=0.08
            # )


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

    # Clean up OpenCV windows
    cv2.destroyAllWindows()
    
    print("\nDemo completed!")
    print("Key takeaways:")
    print("- Ghost vectors don't block rangefinder rays")
    print("- Scene graph visualization has zero physics overhead")
    print("- Perfect for dynamic navigation displays")
    print("- Real-time rangefinder visualization shows obstacle proximity")
