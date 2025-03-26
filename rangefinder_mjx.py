import jax
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx

# Create a simplified MJCF model keeping only necessary parts for rangefinder
xml = """ 
<mujoco model="sensor">
  <asset>
    <material name="material"/>
  </asset>
  <worldbody>
    <!-- tree 0 -->
    <body name="body0" pos="1 2 3">
      <joint name="hinge0" type="hinge" axis="1 0 0"/>
      <geom size="0.1" material="material"/>
      <site name="site_rangefinder0" pos="-1e-3 0 0.2"/>
      <site name="site_rangefinder1" pos="-1e-3 0 0.175"/>
    </body>

    <!-- body for rangefinder -->
    <body name="body_rangefinder" pos="1 2 4">
      <geom size="0.01" material="material"/>
    </body>

  </worldbody>

  <actuator>
    <motor name="motor0" joint="hinge0" ctrlrange="-1 1" gear="10"
      ctrllimited="true"/>
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

# Transfer model and data to MJX (GPU/TPU)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# JIT-compile the forward function for better performance
jit_forward = jax.jit(mjx.forward)

# Step the simulation to compute sensor readings
mjx_data = jit_forward(mjx_model, mjx_data)

# Transfer data back to CPU for printing and visualization
mj_data = mjx.get_data(mj_model, mjx_data)

# Print the rangefinder reading
print(f"Rangefinder0 reading: {mj_data.sensor('rangefinder0').data.item():.4f} meters")
print(f"Rangefinder1 reading: {mj_data.sensor('rangefinder1').data.item():.4f} meters")

# Get the rangefinder site position and orientation
site_id = mj_model.site('site_rangefinder0').id
site_pos = mj_data.site_xpos[site_id].copy()
site_mat = mj_data.site_xmat[site_id].reshape(3, 3)
site_x_axis = site_mat[:, 0]  # First column is the x-axis

# Create visualization options (MJX doesn't have its own renderer, so we use MuJoCo's)
scene_option = mujoco.MjvOption()
# Enable rangefinder visualization
scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True

# Create renderer, render one frame
height, width = 480, 640
with mujoco.Renderer(mj_model, height, width) as renderer:
    # Update scene with rangefinder visualization enabled
    renderer.update_scene(mj_data, scene_option=scene_option)

    # Render
    pixels = renderer.render()

    # Display with matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(pixels)
    plt.axis('off')
    plt.title('Rangefinder Visualization with MJX')
    plt.show()