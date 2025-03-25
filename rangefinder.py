import mujoco
import numpy as np
import matplotlib.pyplot as plt

# Create a simple MJCF model with site rotated correctly
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
      <site name="site0" pos=".1 .2 .3"/>
      <body name="body1" pos="0.1 0.2 0.3">
        <joint name="hinge1" type="hinge" axis="0 1 0"/>
        <geom size="0.25"/>
        <site name="site1" pos=".2 .4 .6"/>
      </body>
    </body>

    <!-- body 2 -->
    <body name="body2" pos=".1 .1 .1">
      <joint name="ballquat2" type="ball" pos="0.1 0.1 0.1"/>
      <geom name="geom2" size="1"/>
    </body>

    <!-- body 3 -->
    <body name="body3" pos="-.1 -.1 -.1">
      <joint name="ballquat3" type="ball" pos="0.1 0.2 0.3"/>
      <geom size="1"/>
      <site name="site3"/>
    </body>

    <!-- bodies for camera projection -->
    <body pos="11.1 0 1">
      <geom type="box" size=".1 .6 .375"/>
      <site name="frontorigin" pos="-.1  .6  .375"/>
      <site name="frontcenter" pos="-.1 0 0"/>
    </body>
    <body pos="10 0 0">
      <joint axis="0 0 1" range="-180 180" limited="false"/>
      <geom type="sphere" size=".2" pos="0 0 0.9"/>
      <camera pos="0 0 1" xyaxes="0 -1 0 0 0 1" fovy="41.11209"
              resolution="1920 1200" name="fixedcamera"/>
    </body>

    <!-- body for rangefinder -->
    <body name="body_rangefinder" pos="1 2 4">
      <geom size="0.01" material="material"/>
    </body>

    <!-- bodies for force + torque sensors -->
    <body pos="-1 -1 -1">
      <joint type="hinge" axis="1 0 0"/>
      <joint type="hinge" axis="0 1 0"/>
      <joint type="hinge" axis="0 0 1"/>
      <site name="site_force"/>
      <geom size="0.1"/>
    </body>
    <body pos="-2 -2 -2">
      <joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <joint type="slide" axis="0 0 1"/>
      <site name="site_torque"/>
      <geom size="0.1"/>
    </body>

    <!-- plane and bodies for touch sensors -->
    <geom type="plane" size="2 2 .1" pos="-20 -20 -20"/>
    <body pos="-20 -20 -20">
      <freejoint/>
      <geom type="capsule" fromto="-.5 0 0 .5 0 0" size="0.0125" condim="1"/>
      <site name="touch_sphere" type="sphere" pos="-0.5 0 0" size="0.025"/>
    </body>
    <body pos="-20 -20.25 -20">
      <freejoint/>
      <geom type="capsule" fromto="-.5 0 0 .5 0 0" size="0.0125" condim="3"/>
      <site name="touch_capsule" type="capsule" fromto="0.4 0 0.0 0.5 0 0.0" size="0.025"/>
    </body>
    <body pos="-20 -20.5 -20">
      <freejoint/>
      <geom type="capsule" fromto="-.5 0 0 .5 0 0" size="0.0125" condim="6"/>
      <site name="touch_box" pos="-0.5 0 0" type="box" size="0.025 0.025 0.025"/>
    </body>
    <body pos="-20 -20.75 -20">
      <freejoint/>
      <geom type="capsule" fromto="-.5 0 0 .5 0 0" size="0.0125" condim="3"/>
      <site name="touch_ellipsoid" type="ellipsoid" pos="0.5 0 0.0" size="0.05 0.01 0.02"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor0" joint="hinge0" ctrlrange="-1 1" gear="10"
      ctrllimited="true"/>
    <motor name="motor1" joint="hinge1" ctrlrange="-1 1" gear="10"
      ctrllimited="true"/>
  </actuator>

  <tendon>
    <fixed name="fixed">
      <joint joint="hinge0" coef=".1"/>
      <joint joint="hinge1" coef=".2"/>
    </fixed>
    <spatial name="spatial">
      <site site="site0"/>
      <site site="site1"/>
    </spatial>
  </tendon>

  <sensor>
    <rangefinder name="rangefinder0" site="site_rangefinder0"/>
    <rangefinder name="rangefinder1" site="site_rangefinder1"/>
  </sensor>
</mujoco>
"""

# Create model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Step the simulation to compute sensor readings
mujoco.mj_forward(model, data)

# Print the rangefinder reading
print(f"Rangefinder reading: {data.sensordata} meters")

# Get the rangefinder site position and orientation
site_id = model.site('site_rangefinder0').id
site_pos = data.site_xpos[site_id].copy()
site_mat = data.site_xmat[site_id].reshape(3, 3)
# For visualization purposes, let's look at all three axis directions
site_x_axis = site_mat[:, 0]  # First column is the x-axis
site_y_axis = site_mat[:, 1]  # Second column is the y-axis
site_z_axis = site_mat[:, 2]  # Third column is the z-axis

print(f"Rangefinder site position: {site_pos}")
print(f"Rangefinder site x-axis: {site_x_axis}")
print(f"Rangefinder site y-axis: {site_y_axis}")
print(f"Rangefinder site z-axis: {site_z_axis}")

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
    plt.title('Rangefinder Visualization with Debug Ray')
    plt.show()