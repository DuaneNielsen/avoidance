import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
import mujoco.viewer
import time

"""
MJX Vehicle Simulation - FIXED VERSION
- Proper data synchronization between MJX and MuJoCo viewer
- Correct control input handling
- Fixed sensor visualization
"""

SENSOR_ANGLE_DEGREES = 64
NUM_SENSORS = 32


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


# Generate sensor sites and rangefinders XML
base_sensor_rotation = np.array([0.5, -0.5, 0.5, 0.5])
sensor_site_xml = ""
sensor_rangefinders_xml = ""
rangefinder_angles = np.linspace(start=-SENSOR_ANGLE_DEGREES, stop=SENSOR_ANGLE_DEGREES, num=NUM_SENSORS)

for i, theta in enumerate(rangefinder_angles):
    rf = np.zeros(4)
    sensor_angle = create_rotation_quat([1, 0, 0], theta)
    rf = mul_quat(base_sensor_rotation, sensor_angle)
    sensor_site_xml += f"""
              <site name="site_rangefinder{i}" quat="{rf[0]} {rf[1]} {rf[2]} {rf[3]}" size="0.01" rgba="1 0 0 1"/>
            """
    sensor_rangefinders_xml += f"""
        <rangefinder name="rangefinder{i}" site="site_rangefinder{i}"/>
    """

xml = f"""
<mujoco model="mjx_vehicle_with_terrain">
  <compiler autolimits="true"/>
  <option timestep="0.01"/>

  <asset>
    <hfield name="terrain" nrow="128" ncol="128" size="10 10 2.0 0.1"/>
    <texture name="terrain_texture" type="2d" builtin="checker" 
             width="512" height="512" rgb2="0.2 0.4 0.2" rgb1="0.6 0.8 0.6"/>
    <material name="terrain_material" texture="terrain_texture" texrepeat="16 16" 
              texuniform="true" reflectance="0.3"/>
    <material name="vehicle_material" rgba="0.2 0.8 0.2 1"/>
  </asset>

  <worldbody>
    <light name="top" pos="0 0 8" dir="0 0 -1"/>
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

      <frame pos="0 0.15 0">
        {sensor_site_xml}
      </frame>
    </body>
  </worldbody>

  <actuator>
    <velocity name="drive" site="control_site" kv="8.0" gear="1 0 0 0 0 0" 
              ctrlrange="-2.0 2.0"/>
    <velocity name="steer" joint="rotate_z" kv="5.0" ctrlrange="-3.0 3.0"/>
    <damper name="brake_x" joint="slide_x" kv="40.0" ctrlrange="0 1"/>
    <damper name="brake_y" joint="slide_y" kv="40.0" ctrlrange="0 1"/>
    <damper name="brake_rot" joint="rotate_z" kv="32.0" ctrlrange="0 1"/>
  </actuator>

  <sensor>
    {sensor_rangefinders_xml}
  </sensor>
</mujoco>
"""


def generate_terrain_with_flat_center(nrow, ncol, hills_x=6, hills_y=6,
                                      hill_height=0.6, hill_radius=0.25,
                                      flat_radius=1.5):
    """Generate heightfield terrain with hills and flattened center."""
    terrain = np.zeros((nrow, ncol))
    world_size_x = 20.0
    world_size_y = 20.0

    spacing_x = nrow / (hills_x + 1)
    spacing_y = ncol / (hills_y + 1)
    center_x = nrow / 2
    center_y = ncol / 2

    flat_radius_grid_x = flat_radius * nrow / world_size_x
    flat_radius_grid_y = flat_radius * ncol / world_size_y

    # Generate hills
    for i in range(hills_x):
        for j in range(hills_y):
            hill_center_x = (i + 1) * spacing_x
            hill_center_y = (j + 1) * spacing_y

            dist_from_center = np.sqrt((hill_center_x - center_x) ** 2 + (hill_center_y - center_y) ** 2)
            if dist_from_center < max(flat_radius_grid_x, flat_radius_grid_y) * 1.5:
                continue

            for x in range(nrow):
                for y in range(ncol):
                    dist = np.sqrt((x - hill_center_x) ** 2 + (y - hill_center_y) ** 2)
                    radius = hill_radius * min(spacing_x, spacing_y)

                    if dist < radius:
                        height = hill_height * (np.cos(np.pi * dist / radius) + 1) / 2
                        terrain[x, y] = max(terrain[x, y], height)

    # Flatten center region
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

    return terrain.flatten()


def print_sensor_summary(sensor_readings):
    """Print a compact sensor summary."""
    valid_readings = [r for r in sensor_readings if r > 0 and r < 100]

    if valid_readings:
        min_dist = min(valid_readings)
        avg_dist = np.mean(valid_readings)

        if min_dist < 1.0:
            status = "⚠️  CLOSE"
        elif min_dist < 2.0:
            status = "⚡ OBSTACLE"
        else:
            status = "✅ CLEAR"

        print(f"Sensors: {len(valid_readings)}/{NUM_SENSORS} | {status} | Min={min_dist:.2f}m Avg={avg_dist:.2f}m")
    else:
        print(f"Sensors: 0/{NUM_SENSORS} | No obstacles detected")


class MJXVehicleSimulation:
    def __init__(self):
        # Create MuJoCo model and data
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Generate and set terrain
        nrow, ncol = 128, 128
        terrain_data = generate_terrain_with_flat_center(
            nrow=nrow, ncol=ncol, hills_x=6, hills_y=6,
            hill_height=0.6, hill_radius=0.25, flat_radius=1.5
        )
        self.mj_model.hfield_data[:] = terrain_data

        # Initialize MuJoCo simulation
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Transfer to MJX
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        # JIT compile MJX functions
        self.jit_step = jax.jit(mjx.step)

        # Control state
        self.forward_vel = 0.0
        self.rotate_vel = 0.0
        self.brake_force = 0.0

        # Statistics
        self.step_count = 0
        self.total_distance = 0.0
        self.last_position = np.array([0.0, 0.0])

        print("✅ MJX Vehicle Simulation Initialized")
        print(f"- {NUM_SENSORS} rangefinder sensors")
        print(f"- Terrain: {nrow}x{ncol} heightfield")

    def key_callback(self, keycode):
        if keycode == 32:  # Spacebar - BRAKE
            self.forward_vel = 0.0
            self.rotate_vel = 0.0
            self.brake_force = 1.0
            print("🛑 BRAKING")
        elif keycode == 265:  # Up arrow - forward
            self.forward_vel = 1.0
            self.rotate_vel = 0.0
            self.brake_force = 0.0
            print("⬆️  FORWARD")
        elif keycode == 264:  # Down arrow - backward
            self.forward_vel = -1.0
            self.rotate_vel = 0.0
            self.brake_force = 0.0
            print("⬇️  BACKWARD")
        elif keycode == 263:  # Left arrow - rotate left
            self.rotate_vel = 1.0
            self.brake_force = 0.0
            print("⬅️  ROTATE LEFT")
        elif keycode == 262:  # Right arrow - rotate right
            self.rotate_vel = -1.0
            self.brake_force = 0.0
            print("➡️  ROTATE RIGHT")

    def step_simulation(self):
        """Step the MJX simulation and sync with viewer data."""
        # Update control inputs in MJX
        ctrl = jp.array([
            self.forward_vel,  # Drive
            self.rotate_vel,  # Steer
            self.brake_force,  # X brake
            self.brake_force,  # Y brake
            self.brake_force  # Rotation brake
        ])
        self.mjx_data = self.mjx_data.replace(ctrl=ctrl)

        # Step MJX simulation
        self.mjx_data = self.jit_step(self.mjx_model, self.mjx_data)
        self.step_count += 1

        # Calculate distance traveled
        current_pos = np.array([float(self.mjx_data.qpos[0]), float(self.mjx_data.qpos[1])])
        distance_delta = np.linalg.norm(current_pos - self.last_position)
        self.total_distance += distance_delta
        self.last_position = current_pos

    def sync_to_viewer(self, viewer_data):
        """Copy MJX state to viewer data for visualization."""
        # Convert MJX data back to numpy arrays
        mjx_as_mj = mjx.get_data(self.mj_model, self.mjx_data)

        # Copy essential fields to viewer data
        viewer_data.time = mjx_as_mj.time
        viewer_data.qpos[:] = mjx_as_mj.qpos
        viewer_data.qvel[:] = mjx_as_mj.qvel
        viewer_data.ctrl[:] = mjx_as_mj.ctrl
        viewer_data.sensordata[:] = mjx_as_mj.sensordata

        # Update derived quantities for proper visualization
        mujoco.mj_forward(self.mj_model, viewer_data)

    def print_status(self):
        """Print current simulation status."""
        pos = [float(self.mjx_data.qpos[0]), float(self.mjx_data.qpos[1])]
        angle = float(self.mjx_data.qpos[2])
        speed = np.sqrt(float(self.mjx_data.qvel[0]) ** 2 + float(self.mjx_data.qvel[1]) ** 2)

        # Control status
        if self.brake_force > 0:
            status = "🛑 BRAKE"
        elif self.forward_vel > 0:
            status = "⬆️ FORWARD"
        elif self.forward_vel < 0:
            status = "⬇️ BACKWARD"
        elif self.rotate_vel > 0:
            status = "⬅️ LEFT"
        elif self.rotate_vel < 0:
            status = "➡️ RIGHT"
        else:
            status = "🔴 IDLE"

        print(f"t={float(self.mjx_data.time):.1f}s | pos=({pos[0]:.2f}, {pos[1]:.2f}) | "
              f"angle={np.degrees(angle):.0f}° | speed={speed:.2f}m/s | {status}")

        # Get sensor data from the converted MuJoCo data
        mjx_as_mj = mjx.get_data(self.mj_model, self.mjx_data)
        sensor_readings = [float(mjx_as_mj.sensordata[i]) for i in range(NUM_SENSORS)]
        print_sensor_summary(sensor_readings)
        print("-" * 60)


def main():
    # Create simulation
    sim = MJXVehicleSimulation()

    print("\n🚗 MJX Vehicle Simulation with Rangefinder Array")
    print("Controls:")
    print("  ↑ : Forward")
    print("  ↓ : Backward")
    print("  ← : Rotate Left")
    print("  → : Rotate Right")
    print("  SPACE : BRAKE")
    print("\nPress keys to control the vehicle!")
    print("=" * 60)

    timestep = 1 / 60.0
    last_status_time = 0

    # Launch MuJoCo viewer
    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data,
                                      key_callback=sim.key_callback) as viewer:
        while viewer.is_running():
            start = time.time()

            # Enable rangefinder visualization
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True

            # Step MJX simulation
            sim.step_simulation()

            # Sync MJX results to viewer data
            sim.sync_to_viewer(sim.mj_data)

            # Update viewer
            viewer.sync()

            # Print status every 2 seconds
            current_time = time.time()
            if current_time - last_status_time > 2.0:
                sim.print_status()
                last_status_time = current_time

            # Maintain real-time
            elapsed = time.time() - start
            sleep_time = max(0, timestep - elapsed)
            time.sleep(sleep_time)

    print(f"\n🏁 Simulation complete! Distance traveled: {sim.total_distance:.1f}m")


if __name__ == "__main__":
    main()