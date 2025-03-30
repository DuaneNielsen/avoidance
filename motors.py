import jax
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx
import mediapy as media
from tqdm import trange

xml = """
<mujoco model="thrust_and_rotation_control">

  <asset>
    <texture type="2d" name="grid" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="5 5" texuniform="true" reflectance=".2"/>
  </asset>

  <default>
    <joint damping="0.1"/>
     <geom friction="0.1 0 0" solref="0.01 1" solimp="0.95 0.99 0.001"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <geom name="floor" type="plane" material="grid" size="5 5 0.1" contype="1" conaffinity="1"/>
    
    <body name="vehicle" pos="0 0 0.2">
      <freejoint/>
      <!-- Main body -->
      <geom name="chassis" type="box" size="0.15 1.0 0.03" mass="0.1" rgba="0.2 0.6 0.8 1" contype="1" conaffinity="1"/>
      <!-- Front indicator to see orientation -->
      <geom name="front" type="box" pos="0.12 0 0.02" mass="0.01" size="0.03 0.03 0.01" rgba="1 0.2 0.2 1"/>
      
      <!-- Control site at center of vehicle -->
      <site name="thrust_site" pos="0 0 0" size="0.01"/>
      <site name="rotation_site" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Forward/reverse thrust in body frame -->
    <motor name="thrust" site="thrust_site" gear="1 0 0 0 0 0" ctrlrange="-1 1"/>
    
    <!-- Left/right rotation in body frame -->
    <motor name="rotation" site="rotation_site" gear="0 0 0 0 0 1" ctrlrange="-1 1"/>
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
duration = 2.  # seconds
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
    cam.distance = 8.5
    cam.lookat = np.array([0, 0, 0.5])

    thrust, rotation = 1.0, 0.1

    for i in trange(n_frames):

        mjx_data = mjx_data.replace(ctrl=jax.numpy.array([thrust, rotation]))

        # Run multiple steps between frames
        for _ in range(steps_per_frame):
            mjx_data = jit_step(mjx_model, mjx_data)

        # Get data back to CPU
        mj_data = mjx.get_data(mj_model, mjx_data)

        vehicle_pos = mj_data.body('vehicle').xpos
        cam.lookat = vehicle_pos


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
output_filename = "motors.mp4"
media.write_video(output_filename, frames, fps=framerate)
print(f"Video saved to {output_filename}")
