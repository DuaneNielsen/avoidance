"""
6-Direction Rangefinder Test - Stable XY Movement
"""
import numpy as np
import mujoco
import mujoco.viewer as viewer

xml = """ 
<mujoco model="sensor">
  <asset>
    <material name="material"/>
    <texture name="grid" type="2d" builtin="checker" 
             width="512" height="512" rgb2="0.1 0.3 0.1" rgb1="0.3 0.7 0.3"/>
    <material name="terrain_material" texture="grid" texrepeat="4 4" 
              texuniform="true" reflectance="0.2"/>
  </asset>
  
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    
    <!-- Ground plane -->
    <geom name="ground" type="plane" size="5 5 .1" material="terrain_material"/>
    
    <!-- Main body with XY slide joints - no wobbling -->
    <body name="body0" pos="0 0 1">
      <joint name="slide_x" type="slide" axis="1 0 0" range="-3 3"/>
      <joint name="slide_y" type="slide" axis="0 1 0" range="-3 3"/>
      <geom size="0.1" material="material"/>
      
      <!-- 6 rangefinder sites - different orientations -->
      <site name="site_forward" pos="0 0 0" size="0.02" rgba="1 0 0 1"/>
      <site name="site_backward" pos="0 0 0" euler="0 0 180" size="0.02" rgba="0.5 0 0 1"/>
      <site name="site_left" pos="0 0 0" euler="0 0 -90" size="0.02" rgba="0 1 0 1"/>
      <site name="site_right" pos="0 0 0" euler="0 0 90" size="0.02" rgba="0 0.5 0 1"/>
      <site name="site_up" pos="0 0 0" euler="0 -90 0" size="0.02" rgba="0 0 1 1"/>
      <site name="site_down" pos="0 0 0" euler="0 90 0" size="0.02" rgba="1 1 0 1"/>
    </body>

    <!-- Target objects to detect -->
    <body name="target1" pos="2 0 1">
      <geom size="0.2" material="material" rgba="1 0 0 1"/>
    </body>
    
    <body name="target2" pos="0 2 1">
      <geom size="0.15" material="material" rgba="0 1 0 1"/>
    </body>
    
    <body name="target3" pos="0 0 3">
      <geom size="0.1" material="material" rgba="0 0 1 1"/>
    </body>
  </worldbody>

  <sensor>
    <rangefinder name="rangefinder_forward" site="site_forward"/>
    <rangefinder name="rangefinder_backward" site="site_backward"/>
    <rangefinder name="rangefinder_left" site="site_left"/>
    <rangefinder name="rangefinder_right" site="site_right"/>
    <rangefinder name="rangefinder_up" site="site_up"/>
    <rangefinder name="rangefinder_down" site="site_down"/>
  </sensor>

  <actuator>
    <position name="move_x" joint="slide_x" ctrlrange="-3 3" kp="10"/>
    <position name="move_y" joint="slide_y" ctrlrange="-3 3" kp="10"/>
  </actuator>
</mujoco>
"""

def move_body(model, data, t):
    """Move the main body using actuators - no wobbling"""
    # Simple circular motion using actuators
    radius = 1.5
    speed = 0.02

    target_x = radius * np.cos(speed * t)
    target_y = radius * np.sin(speed * t)

    # Set actuator targets
    data.ctrl[0] = target_x  # X position target
    data.ctrl[1] = target_y  # Y position target

def print_sensor_readings(data):
    """Print all sensor readings"""
    directions = ['Forward', 'Backward', 'Left', 'Right', 'Up', 'Down']
    print(f"Time: {data.time:6.2f}s - Position: ({data.qpos[0]:.2f}, {data.qpos[1]:.2f}, 1.00)")

    for i, direction in enumerate(directions):
        reading = data.sensordata[i]
        status = 'HIT' if reading > 0 else 'MISS'
        print(f"  {direction:8s}: {reading:6.3f}m ({status})")
    print()

def main():
    # Create model and data
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print("6-Direction Rangefinder Test - Stable XY Movement")
    print("Red: Forward, Dark Red: Backward")
    print("Green: Left, Dark Green: Right")
    print("Blue: Up, Yellow: Down")
    print("Using XY slide joints - no wobbling!")
    print()

    # Create visualization options
    scene_option = mujoco.MjvOption()
    # Enable rangefinder visualization
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True

    # Launch viewer
    step = 0
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            # Move the body using actuators
            move_body(model, data, data.time)

            # Step the simulation
            mujoco.mj_step(model, data)

            # Print readings every 30 steps
            if step % 30 == 0:
                print_sensor_readings(data)

            # Update visualization
            v.opt.flags = scene_option.flags
            v.sync()
            step += 1

if __name__ == "__main__":
    main()