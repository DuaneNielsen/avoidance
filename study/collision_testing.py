import mujoco
import mujoco.viewer
import numpy as np
import time
import quaterion as quat

"""
Vehicle with corner collision sensors and a single hill obstacle.
Four rangefinder sensors at vehicle corners detect terrain collision.
Sensors fire rays parallel to ground in XY plane.
"""


# Calculate quaternions for sensors
# Default site orientation: X-axis points up (Z in world), Y-axis points right, Z-axis points forward
# We want: X-axis to point in desired ray direction (forward/backward), parallel to ground

# For front sensors: rotate X-axis to point forward (along body's +X)
front_sensor_quat = quat.pitch(90)  # Rotate X-axis from up to forward
print(f"Front sensor quaternion: {quat.quat_string(front_sensor_quat)}")

# For rear sensors: rotate X-axis to point backward (along body's -X)
rear_sensor_quat = quat.compose(quat.pitch(90), quat.roll(-90))  # First point forward, then flip 180°
print(f"Rear sensor quaternion: {quat.quat_string(rear_sensor_quat)}")

xml = f"""
<mujoco model="vehicle_collision_test">
  <option timestep="0.01"/>

  <asset>
    <!-- Heightfield terrain -->
    <hfield name="terrain" nrow="128" ncol="128" size="10 10 2.0 0.1"/>
    <texture name="terrain_texture" type="2d" builtin="checker" 
             width="512" height="512" rgb2="0.2 0.4 0.2" rgb1="0.6 0.8 0.6"/>
    <material name="terrain_material" texture="terrain_texture" texrepeat="16 16" 
              texuniform="true" reflectance="0.3"/>

    <material name="vehicle_material" rgba="0.2 0.8 0.2 1"/>
    <material name="sensor_material" rgba="1 0.5 0 0.8"/>
  </asset>

  <worldbody>
    <light name="top" pos="0 0 8" dir="0 0 -1"/>

    <!-- Heightfield terrain -->
    <geom name="terrain" type="hfield" hfield="terrain" material="terrain_material"
          friction="0.8 0.1 0.1"/>

    <body name="vehicle" pos="0 0 0.2">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="1.0"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="1.0"/>
      <joint name="rotate_z" type="hinge" axis="0 0 1" damping="0.5"/>

      <geom name="box" type="box" size="0.3 0.15 0.05" material="vehicle_material"
            friction="0.8 0.1 0.1"/>

      <site name="control_site" pos="0 0 0" size="0.02" rgba="1 0 0 1"/>

      <!-- Corner sensors - positioned at vehicle corners, oriented to fire along the x and y axis -->
      <!-- Front-left corner sensor (fires forward) -->
          <site name="sensor_fl" pos="-0.4 0.25 0" size="0.02" rgba="1 0 0 1" 
                quat="{quat.quat_string(front_sensor_quat)}"/>
    
          <!-- Front-right corner sensor (fires forward) -->
          <site name="sensor_fr" pos="-0.4 -0.25 0" size="0.02" rgba="1 0 0 1" 
                quat="{quat.quat_string(front_sensor_quat)}"/>
    
          <!-- Rear-left corner sensor (fires backward) -->
          <site name="sensor_rl" pos="0.4 -0.25 0" size="0.02" rgba="0 0 1 1" 
                quat="{quat.quat_string(rear_sensor_quat)}"/>
    
          <!-- Rear-right corner sensor (fires backward) -->
          <site name="sensor_rr" pos="-0.4 -0.25 0" size="0.02" rgba="0 0 1 1" 
                quat="{quat.quat_string(rear_sensor_quat)}"/>

      <!-- Visual indicators -->
      <geom name="front_indicator" type="sphere" size="0.03" pos="0.25 0 0" 
            rgba="1 0 0 1" contype="0" conaffinity="0"/>

      <!-- Corner visual markers -->
      <geom name="corner_fl" type="sphere" size="0.02" pos="0.3 0.15 0" 
            material="sensor_material" contype="0" conaffinity="0"/>
      <geom name="corner_fr" type="sphere" size="0.02" pos="0.3 -0.15 0" 
            material="sensor_material" contype="0" conaffinity="0"/>
      <geom name="corner_rl" type="sphere" size="0.02" pos="-0.3 0.15 0" 
            material="sensor_material" contype="0" conaffinity="0"/>
      <geom name="corner_rr" type="sphere" size="0.02" pos="-0.3 -0.15 0" 
            material="sensor_material" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Drive actuators -->
    <velocity name="drive" site="control_site" kv="8.0" gear="1 0 0 0 0 0" 
              ctrlrange="-2.0 2.0"/>
    <velocity name="steer" joint="rotate_z" kv="5.0" ctrlrange="-3.0 3.0"/>

    <!-- Brake actuators -->
    <damper name="brake_x" joint="slide_x" kv="40.0" ctrlrange="0 1"/>
    <damper name="brake_y" joint="slide_y" kv="40.0" ctrlrange="0 1"/>
    <damper name="brake_rot" joint="rotate_z" kv="32.0" ctrlrange="0 1"/>
  </actuator>

  <sensor>
    <!-- Corner rangefinder sensors -->
    <rangefinder name="range_fl" site="sensor_fl"/>
    <rangefinder name="range_fr" site="sensor_fr"/>
    <rangefinder name="range_rl" site="sensor_rl"/>
    <rangefinder name="range_rr" site="sensor_rr"/>
  </sensor>

</mujoco>
"""


def generate_single_hill_terrain(nrow, ncol, hill_x=2.0, hill_y=0.0,
                                 hill_radius=1.0, hill_height=0.8):
    """
    Generate heightfield with a single hill at specified location.

    Args:
        nrow, ncol: Heightfield grid dimensions
        hill_x, hill_y: Hill center in world coordinates
        hill_radius: Hill radius in world units
        hill_height: Maximum height (0-1 range)
    """
    terrain = np.zeros((nrow, ncol))

    # World size from XML: size="10 10 2.0 0.1" means 20x20 world units
    world_size_x = 20.0
    world_size_y = 20.0

    # Convert world coordinates to grid coordinates
    grid_center_x = (hill_x + world_size_x / 2) * nrow / world_size_x
    grid_center_y = (hill_y + world_size_y / 2) * ncol / world_size_y
    grid_radius = hill_radius * nrow / world_size_x

    # Create the single hill
    for x in range(nrow):
        for y in range(ncol):
            # Distance from hill center
            dist = np.sqrt((x - grid_center_x) ** 2 + (y - grid_center_y) ** 2)

            # Smooth circular hill (cosine falloff)
            if dist < grid_radius:
                height = hill_height * (np.cos(np.pi * dist / grid_radius) + 1) / 2
                terrain[x, y] = height

    print(f"Generated single hill at ({hill_x:.1f}, {hill_y:.1f})")
    print(f"Hill radius: {hill_radius:.1f} world units")
    print(f"Hill height: {hill_height:.1f} (normalized)")
    print(f"Terrain height range: {terrain.min():.3f} to {terrain.max():.3f}")

    return terrain.flatten()


# Create model
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Generate terrain with single hill in front of vehicle
nrow, ncol = 128, 128
terrain_data = generate_single_hill_terrain(
    nrow=nrow,
    ncol=ncol,
    hill_x=2.0,  # 2 units in front of vehicle
    hill_y=0.0,  # Centered on Y axis
    hill_radius=1.0,  # 1 unit radius
    hill_height=0.8  # 80% of max height
)

# Set heightfield data
model.hfield_data[:] = terrain_data

timestep = 1 / 60.0

# Control inputs
forward_vel = 0.0
rotate_vel = 0.0
brake_force = 0.0

# Data recording
sensor_data_history = []
time_history = []


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


print("Vehicle Collision Detection Test")
print("=" * 50)
print("Controls:")
print("  ↑ : Forward")
print("  ↓ : Backward")
print("  ← : Rotate Left")
print("  → : Rotate Right")
print("  SPACE : BRAKE")
print()
print("Features:")
print("- Single hill placed 2 units in front of vehicle")
print("- Four corner sensors firing rays parallel to ground")
print("- Front sensors: X-axis rotated -90° around Y (pointing forward)")
print("- Rear sensors: X-axis rotated -90° around Y, then 180° around Z (pointing backward)")
print("- Orange spheres mark sensor positions")
print("- Watch sensor readings as vehicle approaches hill")
print("=" * 50)

frame_count = 0
last_print_time = 0

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        start = time.time()

        # Enable visualization flags
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        # Drive controls
        data.ctrl[0] = forward_vel
        data.ctrl[1] = rotate_vel

        # Brake controls
        data.ctrl[2] = brake_force
        data.ctrl[3] = brake_force
        data.ctrl[4] = brake_force

        mujoco.mj_step(model, data)

        # Read sensor data
        sensor_readings = {
            'fl': data.sensor('range_fl').data[0],
            'fr': data.sensor('range_fr').data[0],
            'rl': data.sensor('range_rl').data[0],
            'rr': data.sensor('range_rr').data[0]
        }

        # Print sensor data every second
        current_time = data.time
        if current_time - last_print_time >= 1.0:
            vehicle_pos = data.qpos[:2]
            vehicle_rot = data.qpos[2]

            print(f"\nTime: {current_time:.1f}s")
            print(f"Vehicle pos: ({vehicle_pos[0]:.2f}, {vehicle_pos[1]:.2f}), rot: {vehicle_rot:.2f} rad")
            print(f"Sensor readings:")
            print(f"  Front-left:  {sensor_readings['fl']:.3f}m {'HIT' if sensor_readings['fl'] > 0 else 'MISS'}")
            print(f"  Front-right: {sensor_readings['fr']:.3f}m {'HIT' if sensor_readings['fr'] > 0 else 'MISS'}")
            print(f"  Rear-left:   {sensor_readings['rl']:.3f}m {'HIT' if sensor_readings['rl'] > 0 else 'MISS'}")
            print(f"  Rear-right:  {sensor_readings['rr']:.3f}m {'HIT' if sensor_readings['rr'] > 0 else 'MISS'}")

            # Check for collision (any front sensor detecting close obstacle)
            front_collision = (sensor_readings['fl'] > 0 and sensor_readings['fl'] < 0.5) or \
                              (sensor_readings['fr'] > 0 and sensor_readings['fr'] < 0.5)
            if front_collision:
                print("⚠️  COLLISION WARNING - Front sensors detecting obstacle!")

            last_print_time = current_time

        viewer.sync()

        curr = time.time()
        while curr - start < timestep:
            time.sleep(0.001)
            curr = time.time()