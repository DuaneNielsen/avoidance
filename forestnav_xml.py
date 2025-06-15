import mujoco
import mujoco.viewer
import numpy as np
import time

xml = """
<mujoco model="box_2d_body_frame">
  <option timestep="0.01"/>

  <asset>
    <material name="box_material" rgba="0.2 0.8 0.2 1"/>
    <material name="floor_material" rgba="0.5 0.5 0.5 1"/>
  </asset>

  <worldbody>
    <light name="top" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="floor_material"/>

    <body name="box_body" pos="0 0 0.05">
      <!-- XY translation joints -->
      <joint name="slide_x" type="slide" axis="1 0 0"/>
      <joint name="slide_y" type="slide" axis="0 1 0"/>
      <!-- Rotation joint -->
      <joint name="rot_joint" type="hinge" axis="0 0 1"/>

      <!-- Rectangular box geometry -->
      <geom name="box" type="box" size="0.15 0.08 0.05" material="box_material"/>

      <!-- Site for body-frame velocity control -->
      <site name="velocity_site" pos="0 0 0" size="0.02" rgba="1 0 0 1"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Forward/backward velocity control in body frame -->
    <velocity name="body_y" site="velocity_site" kv="1.0" gear="0 1 0 0 0 0" ctrlrange="-0.5 0.5"/>
    <!-- Angular velocity control around Z axis in body frame -->
    <velocity name="angular_velocity" joint="rot_joint" kv="1.0" ctrlrange="-0.8 0.8"/>
  </actuator>
</mujoco>
"""

# Create model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("Rectangular Box with Slow 2D Body-Frame Control")
print("DOFs: X translation, Y translation, Z rotation")
print("Controls:")
print("  ctrl[0] = Forward/backward velocity in body frame (m/s)")
print("  ctrl[1] = Angular velocity around Z axis (rad/s)")
print()
print("Motion sequence (slow speed):")
print("  0-4s: Move forward slowly")
print("  4-8s: Turn left while moving forward")
print("  8-12s: Move backward slowly")
print("  12-16s: Turn right while moving forward")
print("  16-24s: Circle maneuver")
print("  24s+: Stop")
print()

# Reset simulation
mujoco.mj_resetData(model, data)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()

    while viewer.is_running():
        step_start = time.time()
        current_time = time.time() - start_time

        # Slow body-frame control sequence
        if current_time < 4.0:
            # Move forward slowly in body frame
            data.ctrl[0] = 0.3  # slow forward velocity
            data.ctrl[1] = 0.0  # no rotation
        elif current_time < 8.0:
            # Turn left while moving forward slowly
            data.ctrl[0] = 0.2  # slow forward velocity
            data.ctrl[1] = 0.4  # slow left turn
        elif current_time < 12.0:
            # Move backward slowly
            data.ctrl[0] = -0.25  # slow backward velocity
            data.ctrl[1] = 0.0  # no rotation
        elif current_time < 16.0:
            # Turn right while moving forward slowly
            data.ctrl[0] = 0.2  # slow forward velocity
            data.ctrl[1] = -0.4  # slow right turn
        elif current_time < 24.0:
            # Slow circle maneuver
            data.ctrl[0] = 0.25  # slow constant forward speed
            data.ctrl[1] = 0.5  # slow constant angular velocity
        else:
            # Stop all motion
            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.0

        # Step simulation
        mujoco.mj_step(model, data)

        # Print status every 2 seconds
        if int(current_time / 2) != int((current_time - model.opt.timestep) / 2):
            # Get box position and angle
            x_pos = data.qpos[0]  # X position (slide_x)
            y_pos = data.qpos[1]  # Y position (slide_y)
            angle = data.qpos[2]  # Z rotation (rot_joint)

            print(f"t={current_time:.1f}s: pos=({x_pos:.2f}, {y_pos:.2f}), "
                  f"heading={np.degrees(angle):.1f}°")

        # Sync with viewer
        viewer.sync()

        # Control frame rate
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("Simulation complete!")