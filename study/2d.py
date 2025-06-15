"""
MuJoCo 2D Body-Frame Velocity Control - Minimal Reference Example

PURPOSE:
Demonstrates 2DOF velocity control in body frame (forward/backward + rotation) using
site-based transmission with gear vectors for tank-like or car-like motion control.

KEY MUJOCO CONCEPTS:
- Joint types: slide (translation) vs hinge (rotation) vs freejoint (6DOF)
- Site transmission: Uses site orientation + gear vector for body-frame forces
- Actuator types: velocity actuators with site/joint transmission
- Body-frame vs world-frame: Forces applied relative to object's orientation
- Friction: Light joint damping + surface friction for realistic motion

CONTROL SETUP:
- Site orientation: quat="0.707 0 0 0.707" rotates site 90° (X-axis = forward)
- Gear vectors: "1 0 0 0 0 0" = force in site's X-axis (forward/backward)
- Transmission types: Site (body-frame) vs Joint (direct joint control)

CONTROL INTERFACE:
- data.ctrl[0] > 0: Move forward in body frame (site's +X direction)
- data.ctrl[0] < 0: Move backward in body frame (site's -X direction)
- data.ctrl[1] > 0: Turn left (positive angular velocity around Z)
- data.ctrl[1] < 0: Turn right (negative angular velocity around Z)

FRICTION EFFECTS:
- Light joint damping: Minimal velocity-dependent resistance
- Light surface friction: Subtle ground interaction effects
- Helps with settling but preserves momentum demonstration
- More realistic than frictionless but doesn't dominate dynamics

PHYSICS INSIGHTS:
- Velocity control ≠ Position control (momentum effects prevent exact reversibility)
- Light friction improves realism without hiding momentum effects
- Real vehicles need position feedback + brakes for precise positioning
- Body-frame control gives intuitive "drive forward/turn" interface

APPLICATIONS:
- Mobile robot control, vehicle simulation, drone control
- Any system needing intuitive forward/backward + turn control
- Cases where body-frame forces are more natural than world-frame forces
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

xml = """
<mujoco model="body_frame_control">
  <option timestep="0.01"/>
  
  <asset>
    <material name="box_material" rgba="0.2 0.8 0.2 1"/>
    <material name="floor_material" rgba="0.5 0.5 0.5 1"/>
  </asset>

  <worldbody>
    <light name="top" pos="0 0 3" dir="0 0 -1"/>
    <!-- Floor with light surface friction -->
    <geom name="floor" type="plane" size="5 5 0.1" material="floor_material" 
          friction="0.3 0.05 0.05"/>
    
    <body name="vehicle" pos="0 0 0.05">
      <!-- Constrain to 2D motion with light joint damping -->
      <joint name="slide_x" type="slide" axis="1 0 0" damping="0.1"/>
      <joint name="slide_y" type="slide" axis="0 1 0" damping="0.1"/>
      <joint name="rot_joint" type="hinge" axis="0 0 1" damping="0.05"/>
      
      <!-- Vehicle geometry with light surface friction -->
      <geom name="box" type="box" size="0.15 0.08 0.05" material="box_material"
            friction="0.3 0.05 0.05"/>
      
      <!-- Site for body-frame control: quat rotates X-axis to point forward -->
      <site name="control_site" pos="0 0 0" size="0.02" rgba="1 0 0 1" 
            quat="0.707 0 0 0.707"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Body-frame forward/backward: gear="1 0 0 0 0 0" = force in site's X-axis -->
    <velocity name="drive" site="control_site" kv="2.0" gear="1 0 0 0 0 0" 
              ctrlrange="-0.5 0.5"/>
    
    <!-- Direct angular velocity control around Z-axis -->
    <velocity name="steer" joint="rot_joint" kv="2.0" ctrlrange="-0.8 0.8"/>
  </actuator>
</mujoco>
"""

def main():
    # Create model and simulation
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print("Body-Frame Control Demo with Light Friction:")
    print("  ctrl[0] = forward/backward velocity (m/s)")
    print("  ctrl[1] = angular velocity (rad/s)")
    print("  Light friction: Subtle realism without overwhelming dynamics")
    print("  Motion: forward → brake → backward → brake → turn")
    print(f"  Joint damping: {model.dof_damping}")
    print(f"  Surface friction: {model.geom_friction[0]} (sliding, torsion, rolling)")

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()

        while viewer.is_running():
            current_time = time.time() - start_time

            # Motion sequence - same as original, light friction preserves behavior
            if current_time < 2.0:
                # Drive forward
                data.ctrl[0] = 0.3   # forward velocity
                data.ctrl[1] = 0.0   # no turn
            elif current_time < 3.0:
                # Brake (light friction helps slightly with settling)
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0
            elif current_time < 5.0:
                # Drive backward
                data.ctrl[0] = -0.3  # backward velocity
                data.ctrl[1] = 0.0
            elif current_time < 9.0:
                # Brake
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0
            elif current_time < 18.0:
                # Turn while driving
                data.ctrl[0] = 0.3   # forward
                data.ctrl[1] = 0.5   # left turn
            else:
                # Stop
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0

            # Step simulation and sync viewer
            mujoco.mj_step(model, data)
            viewer.sync()

            # Print status every 2 seconds
            if int(current_time) % 2 == 0 and int(current_time) != int(current_time - model.opt.timestep):
                speed = np.sqrt(data.qvel[0]**2 + data.qvel[1]**2)
                pos = data.qpos[:2]
                print(f"t={current_time:.0f}s: pos=({pos[0]:.3f}, {pos[1]:.3f}), speed={speed:.3f}m/s")

            # Real-time control
            time.sleep(max(0, model.opt.timestep - (time.time() - start_time - current_time)))

if __name__ == "__main__":
    main()