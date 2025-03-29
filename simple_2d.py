import jax
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx
import mediapy as media

xml = """
<mujoco model="simple_2d">
  <compiler autolimits="true"/>

  <asset>
    <material name="body_material" rgba="0.2 0.8 0.2 1"/>
    <material name="obstacle_material" rgba="0.8 0.2 0.2 1"/>
    <material name="floor_material" rgba="0.3 0.3 0.3 1"/>
  </asset>

  <option timestep="0.02">
    <flag contact="disable" />
  </option>

  <default>
    <geom type="box" pos="0 0 0" size=".1 .15 .05" mass="0.1"/>
    <joint damping="0.25" stiffness="0.0"/>
  </default>
  
  <worldbody>
"""

xml += """
    <site name="origin"/>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>

    <!-- stacked joint: hinge + slide -->
    <body pos="0.0 0 0" name="body">
      <joint name="slidex" type="slide" axis="1. 0. 0." range="0 1"/>
      <joint name="slidey" type="slide" axis="0. 1. 0." range="0 1"/>
      <joint name="hinge0" type="hinge" axis="0 0 1."/>
      <geom/>
    </body>
"""

# Define obstacles - each is [x, y, radius]
obstacles = [
    [0.2, 0.3, 0.1],
    [-0.5, 0.7, 0.07],
    [0.7, -0.5, 0.06]
]


for i, (x, y, radius) in enumerate(obstacles):
    xml += f"""
    <!-- Obstacle {i} -->
    <geom name="obstacle_{i}" type="sphere" pos="{x} {y} 0" 
          size="{radius}" contype="1" conaffinity="1" material="obstacle_material"/>
"""


xml += """    
  </worldbody>


  <actuator>
    <position name="rotation_control" joint="hinge0" kp="10" ctrlrange="-3.14 3.14"/>
    <position name="x_control" joint="slidex" kp="10" ctrlrange="-1 1"/>
    <position name="y_control" joint="slidey" kp="10" ctrlrange="-1 1"/>
  </actuator>

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

    target_position, x, y = 0., 0., 0.

    for i in range(n_frames * 3):

        if 0 <= i < n_frames:
            target_position = 2 * np.pi * (i / n_frames)

        if n_frames <= i < n_frames * 2:
            x = (i - n_frames) / n_frames

        if n_frames * 2 <= i < n_frames * 3:
            y = (i - 2 * n_frames) / n_frames

        mjx_data = mjx_data.replace(ctrl=jax.numpy.array([target_position, x, y]))

        # Run multiple steps between frames
        for _ in range(steps_per_frame):
            mjx_data = jit_step(mjx_model, mjx_data)

        # Get data back to CPU
        mj_data = mjx.get_data(mj_model, mjx_data)

        # Record data
        # time_points.append(mj_data.time)
        # rangefinder0_data.append(mj_data.sensor('rangefinder0').data.item())
        # rangefinder1_data.append(mj_data.sensor('rangefinder1').data.item())
        # joint_angle_data.append(mj_data.qpos[0])

        # Render the frame
        renderer.update_scene(mj_data, camera=cam, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)

# Create video file
output_filename = "simple_2d.mp4"
media.write_video(output_filename, frames, fps=framerate)
print(f"Video saved to {output_filename}")

# Plot rangefinder readings and joint position
plt.figure(figsize=(12, 8))
#
# # Plot rangefinder readings
# plt.subplot(2, 1, 1)
# plt.plot(time_points, rangefinder0_data, label='Rangefinder 0 (Red)', color='red', linewidth=2)
# plt.plot(time_points, rangefinder1_data, label='Rangefinder 1 (Blue)', color='blue', linewidth=2)
# plt.xlabel('Time (s)')
# plt.ylabel('Distance (m)')
# plt.title('Rangefinder Readings')
# plt.legend()
# plt.grid(True)
#
# # Plot joint angle
# plt.subplot(2, 1, 2)
# plt.plot(time_points, joint_angle_data, label='Joint Angle', color='green', linewidth=2)
# plt.xlabel('Time (s)')
# plt.ylabel('Angle (rad)')
# plt.title('Joint Angle')
# plt.grid(True)
#
# plt.tight_layout()
# plt.savefig('rangefinder_data.png')
# plt.show()
#
# print("Simulation complete!")