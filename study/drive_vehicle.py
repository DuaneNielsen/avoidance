import mujoco
import mujoco.viewer
import time


xml = """
<mujoco model="vehicle_with_brakes">
  <option timestep="0.01"/>

  <asset>
    <material name="vehicle_material" rgba="0.2 0.8 0.2 1"/>
    <material name="floor_material" rgba="0.5 0.5 0.5 1"/>
  </asset>

  <worldbody>
    <light name="top" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" material="floor_material" 
          friction="0.8 0.1 0.1"/>

    <body name="vehicle" pos="0 0 0.1">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="1.0"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="1.0"/>
      <joint name="rotate_z" type="hinge" axis="0 0 1" damping="0.5"/>

      <geom name="box" type="box" size="0.3 0.15 0.05" material="vehicle_material"
            friction="0.8 0.1 0.1"/>

      <site name="control_site" pos="0 0 0" size="0.02" rgba="1 0 0 1" />

      <!-- Visual indicator for front -->
      <geom name="front_indicator" type="sphere" size="0.03" pos="0.25 0 0" 
            rgba="1 0 0 1"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Drive actuators -->
    <velocity name="drive" site="control_site" kv="8.0" gear="1 0 0 0 0 0" 
              ctrlrange="-2.0 2.0"/>
    <velocity name="steer" joint="rotate_z" kv="5.0" ctrlrange="-3.0 3.0"/>

    <!-- Brake actuators - dampers that resist motion -->
    <damper name="brake_x" joint="slide_x" kv="40.0" ctrlrange="0 1"/>
    <damper name="brake_y" joint="slide_y" kv="40.0" ctrlrange="0 1"/>
    <damper name="brake_rot" joint="rotate_z" kv="32.0" ctrlrange="0 1"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

timestep = 1 / 60.0

# Control inputs
forward_vel = 0.0
rotate_vel = 0.0
brake_force = 0.0


def key_callback(keycode):
    global forward_vel, rotate_vel, brake_force

    if keycode == 32:  # Spacebar - BRAKE
        forward_vel = 0.0
        rotate_vel = 0.0
        brake_force = 1.0  # Full brake
        print("BRAKING")
    elif keycode == 265:  # Up arrow - forward
        forward_vel = 1.0
        rotate_vel = 0.0
        brake_force = 0.0
        print("FORWARD")
    elif keycode == 264:  # Down arrow - backward
        forward_vel = -1.0
        rotate_vel = 0.0
        brake_force = 0.0
        print("BACKWARD")
    elif keycode == 263:  # Left arrow - rotate left
        rotate_vel = 1.0
        brake_force = 0.0
        print("ROTATE LEFT")
    elif keycode == 262:  # Right arrow - rotate right
        rotate_vel = -1.0
        brake_force = 0.0
        print("ROTATE RIGHT")


print("Vehicle with Damper Brakes")
print("Controls:")
print("  ↑ : Forward")
print("  ↓ : Backward")
print("  ← : Rotate Left")
print("  → : Rotate Right")
print("  SPACE : BRAKE (stops all movement)")
print()

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        start = time.time()

        # Drive controls
        data.ctrl[0] = forward_vel  # Drive
        data.ctrl[1] = rotate_vel  # Steer

        # Brake controls (dampers resist velocity)
        data.ctrl[2] = brake_force  # X brake
        data.ctrl[3] = brake_force  # Y brake
        data.ctrl[4] = brake_force  # Rotation brake

        mujoco.mj_step(model, data)
        viewer.sync()

        curr = time.time()
        while curr - start < timestep:
            time.sleep(0.001)
            curr = time.time()