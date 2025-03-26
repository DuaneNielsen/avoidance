import jax
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx
import mediapy as media

xml = """ 
<mujoco model="rangefinder_example">
  <option timestep="0.002"/>

  <visual>
    <global offheight="640" offwidth="480"/>
    <scale contactwidth="0.05" contactheight="0.05" forcewidth="0.05"/>
  </visual>

  <asset>
    <material name="body_material" rgba="0.8 0.2 0.2 1"/>
    <material name="target_material" rgba="0.2 0.8 0.2 1"/>
    <material name="floor_material" rgba="0.3 0.3 0.3 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom type="plane" pos="0 0 0" size="5 5 0.01" material="floor_material"/>

    <!-- Main body with rangefinders -->
    <body name="body0" pos="0 0 0.5">
      <joint name="hinge0" type="hinge" axis="0 0 1" damping="0.1"/>
      <geom type="box" size="0.3 0.05 0.03" material="body_material"/>
      <site name="site_rangefinder0" pos="0.3 0 0" size="0.02" rgba="1 0 0 1" zaxis="1 0 0"/>
      <site name="site_rangefinder1" pos="-0.3 0 0" size="0.02" rgba="0 0 1 1" zaxis="-1 0 0"/>
    </body>

    <!-- Fixed target bodies for rangefinders to detect -->
    <body name="target1" pos="1.0 0 0.5">
      <geom type="box" size="0.05 0.05 0.05" material="target_material"/>
    </body>

    <body name="target2" pos="-1.0 0 0.5">
      <geom type="box" size="0.05 0.05 0.05" material="target_material"/>
    </body>
  </worldbody>

  <actuator>
    <position name="position_control" joint="hinge0" kp="10" ctrlrange="-3.14 3.14"/>
  </actuator>

  <sensor>
    <rangefinder name="rangefinder0" site="site_rangefinder0"/>
    <rangefinder name="rangefinder1" site="site_rangefinder1"/>
  </sensor>
</mujoco>
"""

# Create MuJoCo model and data
mj_model = mujoco.MjModel.from_xml_string(xml)
mj_data = mujoco.MjData(mj_model)

# Transfer model and data to MJX
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# JIT-compile step function
jit_step = jax.jit(mjx.step)

# Simulation parameters
duration = 6.0  # seconds
framerate = 30  # fps
n_frames = int(duration * framerate)
dt = mj_model.opt.timestep
steps_per_frame = max(1, int(1.0 / (framerate * dt)))

# Create visualization options
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
scene_option.frame = mujoco.mjtFrame.mjFRAME_SITE

# Renderer dimensions - match with the offscreen buffer size
width, height = 480, 480  # swapped to match the XML sizes

# Prepare data recording
time_points = []
rangefinder0_data = []
rangefinder1_data = []
joint_angle_data = []

# Reset simulation
mujoco.mj_resetData(mj_model, mj_data)
mjx_data = mjx.put_data(mj_model, mj_data)

# Render and simulate
frames = []
with mujoco.Renderer(mj_model, height, width) as renderer:
    # Position the camera for better view
    cam = mujoco.MjvCamera()
    cam.azimuth = 90
    cam.elevation = -30
    cam.distance = 3.5
    cam.lookat = np.array([0, 0, 0.5])

    for i in range(n_frames):
        # Full rotation over the duration
        target_position = 2 * np.pi * (i / n_frames)

        # Set the control signal for the position actuator
        mjx_data = mjx_data.replace(ctrl=jax.numpy.array([target_position]))

        # Run multiple steps between frames
        for _ in range(steps_per_frame):
            mjx_data = jit_step(mjx_model, mjx_data)

        # Get data back to CPU
        mj_data = mjx.get_data(mj_model, mjx_data)

        # Record data
        time_points.append(mj_data.time)
        rangefinder0_data.append(mj_data.sensor('rangefinder0').data.item())
        rangefinder1_data.append(mj_data.sensor('rangefinder1').data.item())
        joint_angle_data.append(mj_data.qpos[0])

        # Render the frame
        renderer.update_scene(mj_data, camera=cam, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)

# Create video file
output_filename = "rangefinder_simulation.mp4"
media.write_video(output_filename, frames, fps=framerate)
print(f"Video saved to {output_filename}")

# Plot rangefinder readings and joint position
plt.figure(figsize=(12, 8))

# Plot rangefinder readings
plt.subplot(2, 1, 1)
plt.plot(time_points, rangefinder0_data, label='Rangefinder 0 (Red)', color='red', linewidth=2)
plt.plot(time_points, rangefinder1_data, label='Rangefinder 1 (Blue)', color='blue', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Rangefinder Readings')
plt.legend()
plt.grid(True)

# Plot joint angle
plt.subplot(2, 1, 2)
plt.plot(time_points, joint_angle_data, label='Joint Angle', color='green', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Joint Angle')
plt.grid(True)

plt.tight_layout()
plt.savefig('rangefinder_data.png')
plt.show()

print("Simulation complete!")