import mujoco
import numpy as np
import matplotlib.pyplot as plt

# Create a simplified MJCF model keeping only necessary parts for rangefinder
xml = """ 
<mujoco model="sensor">
  <asset>
    <material name="material"/>
  </asset>
  <worldbody>
    <body name="body0" pos="1 2 3">
      <geom size="0.1" material="material"/>
      <site name="site_rangefinder0" pos="-1e-3 0 0.2"/>
    </body>

    <!-- body for rangefinder to detect -->
    <body name="body_rangefinder" pos="1 2 4">
      <geom size="0.01" material="material"/>
    </body>
  </worldbody>

  <sensor>
    <rangefinder name="rangefinder0" site="site_rangefinder0"/>
  </sensor>
</mujoco>
"""

# Create model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Step the simulation to compute sensor readings
mujoco.mj_forward(model, data)

# Print the rangefinder reading
print(f"Rangefinder reading: {data.sensordata[0]:.4f} meters")

# Get the rangefinder site position and orientation
site_id = model.site('site_rangefinder0').id
site_pos = data.site_xpos[site_id].copy()
site_mat = data.site_xmat[site_id].reshape(3, 3)
site_x_axis = site_mat[:, 0]  # First column is the x-axis

# Create visualization options
scene_option = mujoco.MjvOption()
# Enable rangefinder visualization
scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True

# Create renderer, render one frame
height, width = 480, 640
with mujoco.Renderer(model, height, width) as renderer:
    # Update scene with rangefinder visualization enabled
    renderer.update_scene(data, scene_option=scene_option)

    # Render
    pixels = renderer.render()

    # Display with matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(pixels)
    plt.axis('off')
    plt.title('Rangefinder Visualization')
    plt.show()