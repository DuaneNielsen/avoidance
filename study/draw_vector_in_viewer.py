import mujoco
import mujoco.viewer
import numpy as np
import time

xml = """
<mujoco model="controllable_origin_vectors">
  <option timestep="0.01"/>

  <worldbody>
    <light name="top" pos="0 0 3"/>

    <!-- Controllable origin that can move in XY plane -->
    <body name="origin_body" pos="0 0 0.1">
      <joint name="slide_x" type="slide" axis="1 0 0" range="-3 3" damping="5.0"/>
      <joint name="slide_y" type="slide" axis="0 1 0" range="-3 3" damping="5.0"/>

      <!-- Visual marker for origin -->
      <geom name="origin" type="sphere" size="0.08" rgba="1 0 0 1"/>

      <!-- Dynamic vectors as capsules - DISABLED FOR COLLISION -->
      <geom name="vector1" type="capsule" size="0.03" 
            fromto="0 0 0 1 0.5 0.3" rgba="0 0 1 1"
            contype="0" conaffinity="0"/>
      <geom name="vector2" type="capsule" size="0.03" 
            fromto="0 0 0 0.5 -0.8 0.2" rgba="1 0 1 1"
            contype="0" conaffinity="0"/>
    </body>

    <!-- Fixed endpoint targets -->
    <geom name="endpoint1" type="sphere" size="0.05" pos="2 1 0.5" rgba="0 1 0 1"/>
    <geom name="endpoint2" type="sphere" size="0.05" pos="-1.5 -1 0.3" rgba="0 1 1 1"/>

    <!-- Ground plane for reference -->
    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 0.5"/>

    <!-- Test obstacle to verify no collision -->
    <geom name="obstacle" type="box" size="0.3 0.3 0.2" pos="1 0 0.2" rgba="0.8 0.8 0.2 0.7"/>
  </worldbody>

  <actuator>
    <!-- Position actuators with critical damping -->
    <position name="pos_x" joint="slide_x" kp="25" dampratio="1.0" ctrlrange="-3 3"/>
    <position name="pos_y" joint="slide_y" kp="25" dampratio="1.0" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""


def update_vector_geometry(model, data, geom_name, start_pos, end_pos):
    """Update a capsule geometry to point from start_pos to end_pos."""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

    # Calculate vector
    vector = end_pos - start_pos
    length = np.linalg.norm(vector)

    if length > 0:
        # Midpoint of the capsule
        midpoint = (start_pos + end_pos) / 2

        # Update geometry position (relative to parent body)
        model.geom_pos[geom_id] = midpoint - start_pos

        # Update geometry size (radius, half-length)
        model.geom_size[geom_id][1] = length / 2

        # Calculate orientation quaternion
        if length > 1e-6:
            z_axis = np.array([0, 0, 1])
            direction = vector / length

            if np.abs(np.dot(direction, z_axis)) > 0.99999:
                if np.dot(direction, z_axis) > 0:
                    quat = np.array([1, 0, 0, 0])
                else:
                    quat = np.array([0, 1, 0, 0])
            else:
                axis = np.cross(z_axis, direction)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))

                quat = np.zeros(4)
                quat[0] = np.cos(angle / 2)
                quat[1:4] = np.sin(angle / 2) * axis

            model.geom_quat[geom_id] = quat


def main():
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Fixed endpoint positions
    endpoint1 = np.array([2.0, 1.0, 0.5])
    endpoint2 = np.array([-1.5, -1.0, 0.3])

    # Target position for position control
    target_x = 0.0
    target_y = 0.0
    move_step = 1.0

    def key_callback(keycode):
        nonlocal target_x, target_y

        if keycode == 87 or keycode == 265:  # W or Up arrow
            target_y = min(target_y + move_step, 3.0)
            print(f"Target: ({target_x:.2f}, {target_y:.2f})")
        elif keycode == 83 or keycode == 264:  # S or Down arrow
            target_y = max(target_y - move_step, -3.0)
            print(f"Target: ({target_x:.2f}, {target_y:.2f})")
        elif keycode == 65 or keycode == 263:  # A or Left arrow
            target_x = max(target_x - move_step, -3.0)
            print(f"Target: ({target_x:.2f}, {target_y:.2f})")
        elif keycode == 68 or keycode == 262:  # D or Right arrow
            target_x = min(target_x + move_step, 3.0)
            print(f"Target: ({target_x:.2f}, {target_y:.2f})")
        elif keycode == 32:  # Spacebar - Return to center
            target_x = 0.0
            target_y = 0.0
            print("Returning to center (0, 0)")
        elif keycode == 67:  # C - Test collision area
            target_x = 1.0
            target_y = 0.0
            print("Moving to obstacle area - testing collision!")

    print("Position Control with Collision-Free Vectors")
    print("Controls:")
    print("  W/↑ : Move Up")
    print("  S/↓ : Move Down")
    print("  A/← : Move Left")
    print("  D/→ : Move Right")
    print("  SPACE : Return to Center")
    print("  C : Move to obstacle (test collision)")
    print()
    print("Vector capsules have contype=0 conaffinity=0 (no collision)")
    print("Try moving through the yellow obstacle!")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            # Set position targets
            data.ctrl[0] = target_x
            data.ctrl[1] = target_y

            # Step simulation
            mujoco.mj_step(model, data)

            # Get current origin position
            origin_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'origin_body')
            origin_pos = data.xpos[origin_body_id].copy()

            # Update vector geometries
            update_vector_geometry(model, data, 'vector1', origin_pos, endpoint1)
            update_vector_geometry(model, data, 'vector2', origin_pos, endpoint2)

            # Debug: Check for contacts
            if data.ncon > 0:
                print(f"Active contacts: {data.ncon}")
                for i in range(data.ncon):
                    contact = data.contact[i]
                    geom1_name = model.geom(contact.geom1).name or f"geom_{contact.geom1}"
                    geom2_name = model.geom(contact.geom2).name or f"geom_{contact.geom2}"
                    print(f"  Contact {i}: {geom1_name} - {geom2_name}")

            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()