import mujoco
import mujoco.viewer
import numpy as np
import time

# XML model definition with a simple scene
xml_model = """
<mujoco model="color_demo">
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" 
             rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" reflectance=".2"/>
  </asset>

  <worldbody>
    <light name="light" pos="0 0 3"/>
    <geom name="floor" size="2 2 .1" type="plane" material="grid"/>

    <!-- Main geometry that will change color -->
    <body name="colorful_box" pos="0 0 0.5">
      <geom name="box" type="box" size="0.3 0.3 0.3" rgba="1 0 0 1"/>
      <joint name="free_joint" type="free"/>
    </body>

    <!-- Additional geometries for comparison -->
    <body name="sphere1" pos="-1 0 0.3">
      <geom name="sphere1" type="sphere" size="0.2" rgba="0 1 0 1"/>
    </body>

    <body name="cylinder1" pos="1 0 0.3">
      <geom name="cylinder1" type="cylinder" size="0.15 0.3" rgba="0 0 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def rgb_from_hsv(h, s, v):
    """Convert HSV to RGB color space"""
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return (r + m, g + m, b + m)


def main():
    # Load the model
    model = mujoco.MjModel.from_xml_string(xml_model)
    data = mujoco.MjData(model)

    # Find the geometry ID for the box we want to change
    box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")

    print(f"Box geometry ID: {box_geom_id}")
    print("Starting color animation demo...")
    print("The red box will cycle through colors based on time")
    print("Press ESC to exit the viewer")

    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()

        while viewer.is_running():
            # Calculate elapsed time
            elapsed = time.time() - start_time

            # Create time-based color variations

            # Method 1: Rainbow cycle (HSV color space)
            hue = (elapsed * 60) % 360  # Complete cycle every 6 seconds
            saturation = 0.8
            value = 0.9
            r, g, b = rgb_from_hsv(hue, saturation, value)

            # Method 2: Sinusoidal color mixing (uncomment to try)
            # r = 0.5 + 0.5 * np.sin(elapsed * 2)
            # g = 0.5 + 0.5 * np.sin(elapsed * 2 + 2*np.pi/3)
            # b = 0.5 + 0.5 * np.sin(elapsed * 2 + 4*np.pi/3)

            # Method 3: Pulsing brightness (uncomment to try)
            # base_color = [1.0, 0.3, 0.3]  # Red base
            # brightness = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(elapsed * 4))
            # r, g, b = [c * brightness for c in base_color]

            # Update the geometry color
            # The rgba values are stored in model.geom_rgba
            model.geom_rgba[box_geom_id] = [r, g, b, 1.0]  # RGB + Alpha

            # Also demonstrate changing other properties
            # Make the sphere pulse in size
            sphere_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sphere1")
            pulse = 0.15 + 0.1 * (0.5 + 0.5 * np.sin(elapsed * 3))
            model.geom_size[sphere_geom_id] = [pulse, 0, 0]  # Sphere only uses first component

            # Make the cylinder change transparency
            cylinder_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder1")
            alpha = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(elapsed * 1.5))
            model.geom_rgba[cylinder_geom_id][3] = alpha

            # Step the simulation
            mujoco.mj_step(model, data)

            # Sync the viewer
            viewer.sync()

            # Control the frame rate
            time.sleep(1 / 60)  # 60 FPS


if __name__ == "__main__":
    main()