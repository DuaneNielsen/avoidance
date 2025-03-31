import jax
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx

import mediapy as media
from math import sin, cos
from tqdm import trange


xml = """
<mujoco model="simple_2d">
  <compiler autolimits="true"/>
  
  <option integrator="implicitfast"/>

  <asset>
    <material name="body_material" rgba="0.2 0.8 0.2 1"/>
    <material name="obstacle_material" rgba="0.8 0.2 0.2 1"/>
    <material name="floor_material" rgba="0.3 0.3 0.3 1"/>
  </asset>

  <option timestep="0.02">
    <flag contact="disable" />
  </option>

  <default>
    <joint damping="0.25" stiffness="0.0"/>
  </default>
  
  <worldbody>
"""

sensor_angle = 0.6
num_sensors = 128

xml += f"""
    <site name="origin"/>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>

    <!-- stacked joint: hinge + slide -->
    <body pos="0.0 0 0" name="vehicle">
      <joint name="x_joint" type="slide" axis="1. 0. 0." range="-1 1"/>
      <joint name="y_joint" type="slide" axis="0. 1. 0." range="-1 1"/>
      <joint name="rot_joint" type="hinge" axis="0 0 1."/>
      <site name="velocity_site" pos="0 0 0" size="0.01"/>
      <frame pos="0 0.01 0" quat="-1 1 0 0">
      """

rangefinder_angles = np.linspace(start=-sensor_angle, stop=sensor_angle, num=num_sensors)
for i, theta in enumerate(rangefinder_angles):
    xml += f"""
              <site name="site_rangefinder{i}" quat="{cos(theta/2)} 0 {sin(theta/2)} 0" size="0.01" rgba="1 0 0 1"/>
            """

xml += f"""
      </frame>
      <geom type="box" pos="0 0 0" size=".0168 .01 .005" mass="0.1"/>
    </body>
"""


obstacles = []
for x in np.linspace(-1., 1., 5):
    for y in np.linspace(-1., 1, 5):
        if x == 0. and y == 0.:
            continue
        obstacles.append([x, y, 0.07])


for i, (x, y, radius) in enumerate(obstacles):
    xml += f"""
    <!-- Obstacle {i} -->
    <geom name="obstacle_{i}" type="sphere" pos="{x} {y} 0" 
          size="{radius}" contype="1" conaffinity="1" material="obstacle_material"/>
"""


xml += """    
  </worldbody>

  <sensor>
  """

for i in range(num_sensors):
    xml += f"""
        <rangefinder name="rangefinder{i}" site="site_rangefinder{i}"/>
        """

xml += """
    <framequat name="vehicle_quat" objtype="site" objname="velocity_site"/>
  </sensor>

  <actuator>
    
    <!-- Forward/backward velocity control in body frame -->
    <velocity name="body_y" site='velocity_site' kv="1." gear="0 1 0 0 0 0" ctrlrange="-2 2"/>
    
    <!-- Angular velocity control around Z axis in body frame -->
    <velocity name="angular_velocity" joint="rot_joint" kv="1." ctrlrange="-1 1"/>
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
duration = 30.0  # seconds
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
rangefinder_data = []
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
    cam.elevation = -50
    cam.distance = 3.5
    cam.lookat = np.array([0, 0, 0.2])

    target_vel, target_rotation_vel = 0.4, 1.

    for i in trange(n_frames):

        ctrl_rotation_vel = - target_rotation_vel * np.sign(i - n_frames//2)
        ctrl = jax.numpy.array([target_vel, ctrl_rotation_vel])
        mjx_data = mjx_data.replace(ctrl=ctrl)

        # Run multiple steps between frames
        for _ in range(steps_per_frame):
            mjx_data = jit_step(mjx_model, mjx_data)

        # Get data back to CPU
        mj_data = mjx.get_data(mj_model, mjx_data)

        # Record data
        time_points.append(mj_data.time)
        rangefinder_data.append([mj_data.sensor(f'rangefinder{i}').data.item() for i in range(num_sensors)])
        joint_angle_data.append(mj_data.qpos[0])

        # Render the frame
        renderer.update_scene(mj_data, camera=cam, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)

# Create video file
output_filename = "forestnav_v1.mp4"
media.write_video(output_filename, frames, fps=framerate)
print(f"Video saved to {output_filename}")

# Plot rangefinder readings and joint position
plt.figure(figsize=(12, 8))
#
# Plot rangefinder readings
plt.subplot(2, 1, 1)
rangefinder_data = np.array(rangefinder_data)
plt.imshow(rangefinder_data.T)
min_idx = 0  # First element (most negative angle)
center_idx = len(rangefinder_angles) // 2  # Middle element (approximately zero)
max_idx = len(rangefinder_angles) - 1  # Last element (most positive angle)

plt.yticks(
    [min_idx, center_idx, max_idx],
    [f'{rangefinder_angles[min_idx]:.2f}', f'{rangefinder_angles[center_idx]:.2f}', f'{rangefinder_angles[max_idx]:.2f}']
)

plt.xlabel('Time (s)')
plt.ylabel('Angle (m)')
plt.title('Rangefinder Readings')
# plt.legend()
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