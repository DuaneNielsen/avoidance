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

      <!-- Dynamic vector as capsule - LOCAL FRAME (child of body) -->
      <geom name="vector1" type="capsule" size="0.03" 
            fromto="0 0 0 1 0.5 0.3" rgba="0 0 1 1"
            contype="0" conaffinity="0"/>
    </body>

    <!-- Second vector defined in WORLD FRAME (child of worldbody) -->
    <geom name="vector_world" type="capsule" size="0.025" 
          fromto="0 0 0.2 1 0.5 0.5" rgba="1 0 1 1"
          contype="0" conaffinity="0"/>

    <!-- Fixed endpoint target -->
    <geom name="endpoint1" type="sphere" size="0.05" pos="2 1 0.5" rgba="0 1 0 1"/>

    <!-- Vehicle goal position marker -->
    <geom name="goal_marker" type="sphere" size="0.06" pos="-1.5 -1.0 0.2" rgba="1 1 0 1"/>

    <!-- Ground plane for reference -->
    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 0.5"/>
  </worldbody>

  <actuator>
    <!-- Position actuators with critical damping -->
    <position name="pos_x" joint="slide_x" kp="25" dampratio="1.0" ctrlrange="-3 3"/>
    <position name="pos_y" joint="slide_y" kp="25" dampratio="1.0" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""


def update_vector_geometry_local_frame(model, data, geom_name, end_pos, start_pos=None):
    """
    Update a capsule geometry in LOCAL FRAME (child of a body).

    Args:
        model: MuJoCo model
        data: MuJoCo data
        geom_name: Name of the capsule geometry
        end_pos: Target endpoint in world coordinates
        start_pos: Optional start position in world coordinates.
                  If None, uses parent body's origin (recommended for local frame)

    For local frame vectors, start_pos should typically be None to use the parent body's origin.
    This creates vectors that extend FROM the body TO the target.
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

    # If no start_pos specified, get the parent body's world position
    if start_pos is None:
        # Find the parent body of this geometry
        parent_body_id = model.geom_bodyid[geom_id]
        start_pos = data.xpos[parent_body_id].copy()

    # Calculate vector in world coordinates
    vector = end_pos - start_pos
    length = np.linalg.norm(vector)

    if length > 1e-6:  # Avoid division by zero
        # For local frame: capsule extends FROM origin TO target
        # Midpoint in local coordinates = half the vector from origin
        midpoint_local = vector / 2

        # Update geometry position (relative to parent body)
        model.geom_pos[geom_id] = midpoint_local

        # Update geometry size (radius, half-length)
        model.geom_size[geom_id][1] = length / 2

        # Use MuJoCo's built-in function to create quaternion
        direction = vector / length
        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, direction)

        model.geom_quat[geom_id] = quat

        return midpoint_local, length, quat

    return None, 0, None

# Simplified wrapper for the most common use case
def update_local_vector_to_target(model, data, geom_name, target_pos):
    """
    Convenience function: Update local frame vector to point from body origin to target.
    This is the most common use case for local frame vectors.
    """
    return update_vector_geometry_local_frame(model, data, geom_name, target_pos)


def update_vector_geometry_world_frame(model, data, geom_name, start_pos, end_pos):
    """
    Update a capsule geometry in WORLD FRAME (child of worldbody).
    Position is in absolute world coordinates.
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

    # Calculate vector in world coordinates
    vector = end_pos - start_pos
    length = np.linalg.norm(vector)

    if length > 1e-6:  # Avoid division by zero
        # For world frame, capsule center is at midpoint between start and end
        midpoint_world = (start_pos + end_pos) / 2

        # Update geometry position (absolute world coordinates)
        model.geom_pos[geom_id] = midpoint_world

        # Update geometry size (radius, half-length)
        model.geom_size[geom_id][1] = length / 2

        # Use MuJoCo's built-in function to create quaternion
        direction = vector / length
        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, direction)

        model.geom_quat[geom_id] = quat

        return midpoint_world, length, quat

    return None, 0, None


def update_vehicle_to_goal_vector(model, data, vector_geom_name, vehicle_pos, goal_pos):
    """
    Update a world-frame vector to show vehicle position to goal position.
    This demonstrates how to use world-frame vectors for navigation visualization.
    """
    return update_vector_geometry_world_frame(model, data, vector_geom_name, vehicle_pos, goal_pos)


def verify_capsule_geometry(model, data, geom_name, expected_start, expected_end):
    """Verify that the capsule geometry matches expected start/end points."""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

    # Get actual capsule properties
    actual_pos = model.geom_pos[geom_id].copy()
    actual_quat = model.geom_quat[geom_id].copy()
    actual_half_length = model.geom_size[geom_id][1]

    # Calculate what the endpoints should be based on capsule properties
    # Capsule extends along Z-axis by default, so endpoints are at ±half_length in Z
    local_start = np.array([0, 0, -actual_half_length])
    local_end = np.array([0, 0, actual_half_length])

    # Rotate local endpoints by capsule quaternion
    rotated_start = np.zeros(3)
    rotated_end = np.zeros(3)
    mujoco.mju_rotVecQuat(rotated_start, local_start, actual_quat)
    mujoco.mju_rotVecQuat(rotated_end, local_end, actual_quat)

    # Add capsule center position to get world coordinates
    calculated_start = actual_pos + rotated_start
    calculated_end = actual_pos + rotated_end

    return calculated_start, calculated_end


def main():
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Fixed endpoint position for first vector
    endpoint1 = np.array([2.0, 1.0, 0.5])

    # Goal position for vehicle navigation
    goal_position = np.array([-1.5, -1.0, 0.2])

    # Open loop control sequence
    control_sequence = [
        (0.0, 0.0, 2.0),  # Stay at origin for 2 seconds
        (1.0, 0.0, 2.0),  # Move to x=1, y=0 for 2 seconds
        (1.0, 0.5, 2.0),  # Move to x=1, y=0.5 for 2 seconds
        (0.5, 0.5, 2.0),  # Move to x=0.5, y=0.5 for 2 seconds
        (0.0, 0.0, 2.0),  # Return to origin for 2 seconds
    ]

    print("Dual Vector Visualization Test")
    print("=" * 60)
    print(f"Fixed endpoint (green sphere): {endpoint1}")
    print(f"Goal position (yellow sphere): {goal_position}")
    print("\nBlue vector: Local frame (vehicle->endpoint, moves with vehicle)")
    print("Magenta vector: World frame (vehicle->goal, fixed in world)")
    print("=" * 60)

    current_step = 0
    step_start_time = 0
    total_time = 0
    step_printed = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and current_step < len(control_sequence):
            # Get current control targets
            target_x, target_y, duration = control_sequence[current_step]

            # Set position targets
            data.ctrl[0] = target_x
            data.ctrl[1] = target_y

            # Step simulation
            mujoco.mj_step(model, data)
            total_time += model.opt.timestep

            # Get current vehicle/origin position
            origin_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'origin_body')
            vehicle_pos = data.xpos[origin_body_id].copy()

            # Update both vectors
            # 1. Local frame vector (blue): vehicle to fixed endpoint
            update_local_vector_to_target(model, data, 'vector1', endpoint1)

            # 2. World frame vector (magenta): vehicle to goal position
            update_vehicle_to_goal_vector(model, data, 'vector_world', vehicle_pos, goal_position)

            # Verify geometry once per step when settled
            if not step_printed and (total_time - step_start_time) >= 1.5:
                print(f"\nStep {current_step + 1}:")
                print(f"Vehicle position: {vehicle_pos}")

                # Verify local frame vector
                print(f"\nLocal frame vector (blue):")
                print(f"Expected: start={vehicle_pos} -> end={endpoint1}")
                calc_start1, calc_end1 = verify_capsule_geometry(model, data, 'vector1', vehicle_pos, endpoint1)
                print(f"Capsule:  start={calc_start1} -> end={calc_end1}")

                # Verify world frame vector
                print(f"\nWorld frame vector (magenta):")
                print(f"Expected: start={vehicle_pos} -> end={goal_position}")
                calc_start2, calc_end2 = verify_capsule_geometry(model, data, 'vector_world', vehicle_pos,
                                                                 goal_position)
                print(f"Capsule:  start={calc_start2} -> end={calc_end2}")

                # Calculate errors
                start_error1 = np.linalg.norm(calc_start1 - vehicle_pos)
                end_error1 = np.linalg.norm(calc_end1 - endpoint1)
                start_error2 = np.linalg.norm(calc_start2 - vehicle_pos)
                end_error2 = np.linalg.norm(calc_end2 - goal_position)

                print(f"\nErrors:")
                print(f"Local vector:  start_error={start_error1:.6f}, end_error={end_error1:.6f}")
                print(f"World vector:  start_error={start_error2:.6f}, end_error={end_error2:.6f}")

                if all(error < 0.001 for error in [start_error1, end_error1, start_error2, end_error2]):
                    print("✓ PASS - Both vectors match expected behavior")
                else:
                    print("✗ FAIL - One or more vectors don't match expected behavior")

                step_printed = True

            # Check if we should move to next step
            if (total_time - step_start_time) >= duration:
                current_step += 1
                step_start_time = total_time
                step_printed = False

            viewer.sync()
            time.sleep(0.01)

    print("\nTest completed!")
    print("\nKey differences observed:")
    print("- Blue vector (local frame): Moves and rotates with the red vehicle body")
    print("- Magenta vector (world frame): Start point follows vehicle, but stays fixed in world space")
    print("- World frame vectors are ideal for navigation, pathfinding, and global reference lines")
    print("- Local frame vectors are ideal for sensor visualization, robot arm links, vehicle-relative displays")


if __name__ == "__main__":
    main()