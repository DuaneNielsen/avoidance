import mujoco
import mujoco.viewer
import numpy as np
import time
from math import cos, sin

"""
This is a simulation of a 2d vehicle with forward/reverse/brake and rotational steering
The vehicle is located at the origin, and constrained using xy and hinge joints
A heightmap is used for terrain, a grid of small hills, with a flat region around the origin
A single ray is cast in the forward direction of the vehicle to detect terrain
"""

SENSOR_ANGLE = 0.75
NUM_SENSORS = 64



sensor_site_xml = ""
sensor_rangefinders_xml = ""
rangefinder_angles = np.linspace(start=-SENSOR_ANGLE, stop=SENSOR_ANGLE, num=NUM_SENSORS)
for i, theta in enumerate(rangefinder_angles):
    sensor_site_xml += f"""
              <site name="site_rangefinder{i}" quat="{cos(theta / 2)} 0 {sin(theta / 2)} 0" size="0.01" rgba="1 0 0 1"/>
            """
    sensor_rangefinders_xml += f"""
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
  </asset>

  <worldbody>
    <light name="top" pos="0 0 8" dir="0 0 -1"/>

    <!-- Heightfield terrain instead of flat plane -->
    <geom name="terrain" type="hfield" hfield="terrain" material="terrain_material"
          friction="0.8 0.1 0.1"/>

    <body name="vehicle" pos="0 0 0.2">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="1.0"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="1.0"/>
      <joint name="rotate_z" type="hinge" axis="0 0 1" damping="0.5"/>

      <geom name="box" type="box" size="0.3 0.15 0.05" material="vehicle_material"
            friction="0.8 0.1 0.1"/>

      <site name="control_site" pos="0 0 0" size="0.02" rgba="1 0 0 1" 
            quat="0.707 0 0 0.707"/>
       <frame pos="0 0.01 0" quat="-1 1 0 0">
      {sensor_site_xml}
        </frame>
    </body>
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
        {sensor_rangefinders_xml}
    </sensor>

</mujoco>
"""

print(xml)

def generate_terrain_with_flat_center(nrow, ncol, hills_x=6, hills_y=6,
                                      hill_height=0.6, hill_radius=0.25,
                                      flat_radius=1.5):
    """
    Generate heightfield terrain with hills but flatten the center region for vehicle spawn.

    Args:
        nrow, ncol: Heightfield grid dimensions
        hills_x, hills_y: Number of hills in each direction
        hill_height: Maximum height of hills (0-1 range)
        hill_radius: Radius of hills as fraction of spacing
        flat_radius: Radius around center to flatten (in world units)
    """
    terrain = np.zeros((nrow, ncol))

    # World size from XML: size="10 10 2.0 0.1" means 20x20 world units
    world_size_x = 20.0  # Total world size
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
            # Hill center positions
            hill_center_x = (i + 1) * spacing_x
            hill_center_y = (j + 1) * spacing_y

            # Skip hills too close to center
            dist_from_center = np.sqrt((hill_center_x - center_x) ** 2 + (hill_center_y - center_y) ** 2)
            if dist_from_center < max(flat_radius_grid_x, flat_radius_grid_y) * 1.5:
                continue

            # Create circular hill
            for x in range(nrow):
                for y in range(ncol):
                    # Distance from hill center
                    dist = np.sqrt((x - hill_center_x) ** 2 + (y - hill_center_y) ** 2)

                    # Hill radius in grid units
                    radius = hill_radius * min(spacing_x, spacing_y)

                    # Smooth circular hill (cosine falloff)
                    if dist < radius:
                        height = hill_height * (np.cos(np.pi * dist / radius) + 1) / 2
                        terrain[x, y] = max(terrain[x, y], height)

    # Flatten the center region for vehicle spawn
    for x in range(nrow):
        for y in range(ncol):
            # Distance from center
            dist_x = abs(x - center_x)
            dist_y = abs(y - center_y)

            # If within flat radius, set to zero with smooth transition
            if dist_x < flat_radius_grid_x and dist_y < flat_radius_grid_y:
                terrain[x, y] = 0.0

            # Smooth transition at the edge of flat zone
            elif dist_x < flat_radius_grid_x * 1.2 and dist_y < flat_radius_grid_y * 1.2:
                # Gradual transition
                fade_factor = max(
                    (dist_x - flat_radius_grid_x) / (flat_radius_grid_x * 0.2),
                    (dist_y - flat_radius_grid_y) / (flat_radius_grid_y * 0.2)
                )
                fade_factor = np.clip(fade_factor, 0.0, 1.0)
                terrain[x, y] *= fade_factor

    print(f"Generated {hills_x}x{hills_y} hills with flat center")
    print(f"Terrain height range: {terrain.min():.3f} to {terrain.max():.3f}")
    print(f"Flat radius: {flat_radius:.1f} world units")
    print(f"Hill spacing: {spacing_x:.1f} x {spacing_y:.1f} grid units")

    return terrain.flatten()


# Create model
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Generate terrain with flattened center
nrow, ncol = 128, 128
terrain_data = generate_terrain_with_flat_center(
    nrow=nrow,
    ncol=ncol,
    hills_x=6,  # 6 hills in X direction
    hills_y=6,  # 6 hills in Y direction
    hill_height=0.6,  # 60% of max height
    hill_radius=0.25,  # 25% of spacing
    flat_radius=1.5  # 1.5 world units flat around center
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
        brake_force = 1.0  # Full brake
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


print("Vehicle with Hills Terrain and Flat Spawn Area")
print("Controls:")
print("  ↑ : Forward")
print("  ↓ : Backward")
print("  ← : Rotate Left")
print("  → : Rotate Right")
print("  SPACE : BRAKE (stops all movement)")
print()
print("The vehicle starts on flat ground and can drive up/down hills!")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        start = time.time()

        # flags
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True

        # Drive controls
        data.ctrl[0] = forward_vel  # Drive
        data.ctrl[1] = rotate_vel  # Steer

        # Brake controls (dampers resist velocity)
        data.ctrl[2] = brake_force  # X brake
        data.ctrl[3] = brake_force  # Y brake
        data.ctrl[4] = brake_force  # Rotation brake

        mujoco.mj_step(model, data)
        viewer.sync()

        curr = time.time()
        while curr - start < timestep:
            time.sleep(0.001)
            curr = time.time()
