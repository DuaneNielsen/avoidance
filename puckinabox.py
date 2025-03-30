import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import numpy as np

# Import Brax
from brax.io import mjcf
from brax.base import Force, Motion, Transform

# Create an MJCF model for our 2D puck in a box with multiple obstacles
box_size = 2.0
puck_radius = 0.2

# Define obstacles - each is [x, y, radius, color]
obstacles = [
    [0.2, 0.3, 0.3, "0.2 0.7 0.2 1"],  # Green obstacle
    [-0.5, 0.7, 0.25, "0.7 0.2 0.2 1"],  # Red obstacle
    [0.7, -0.5, 0.2, "0.2 0.2 0.7 1"]  # Blue obstacle
]

# Puck starting position
puck_pos = [-0.2, 0.1, 0]

# Begin building the MJCF string - add actuators for control
mjcf_string = """
<mujoco model="2d_puck">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 0" />
  <custom>
    <numeric data="1.0" name="elasticity"/>
    <numeric data="0.0" name="ang_damping"/>
    <numeric data="0.0" name="vel_damping"/>
  </custom>
  <worldbody>
    <!-- The puck -->
    <body name="puck" pos="{0} {1} {2}">
      <joint type="free" name="puck_joint"/>
      <geom name="puck_geom" type="sphere" size="{3}" mass="1" rgba="0.7 0.2 0.8 1"/>
    </body>
""".format(puck_pos[0], puck_pos[1], puck_pos[2], puck_radius)

# Add obstacles
for i, (x, y, radius, color) in enumerate(obstacles):
    mjcf_string += """
    <!-- Obstacle {0} -->
    <geom name="obstacle_{0}" type="cylinder" pos="{1} {2} 0" 
          size="{3} 0.0001" rgba="{4}" contype="1" conaffinity="1"/>
""".format(i, x, y, radius, color)

# Add walls
mjcf_string += """
    <!-- Walls of the box -->
    <geom name="bottom_wall" type="plane" pos="0 0 -0.1" size="{0} {0} 0.1" rgba="0.8 0.8 0.8 1" conaffinity="1"/>
    <geom name="left_wall" type="plane" pos="-{1} 0 0" size="0.1 {0} {0}" rgba="0.8 0.8 0.8 1" conaffinity="1" zaxis="1 0 0"/>
    <geom name="right_wall" type="plane" pos="{1} 0 0" size="0.1 {0} {0}" rgba="0.8 0.8 0.8 1" conaffinity="1" zaxis="-1 0 0"/>
    <geom name="top_wall" type="plane" pos="0 {1} 0" size="{0} 0.1 {0}" rgba="0.8 0.8 0.8 1" conaffinity="1" zaxis="0 -1 0"/>
    <geom name="bottom_wall2" type="plane" pos="0 -{1} 0" size="{0} 0.1 {0}" rgba="0.8 0.8 0.8 1" conaffinity="1" zaxis="0 1 0"/>
  </worldbody>
</mujoco>
""".format(box_size, box_size / 2)

# Load the MJCF model into Brax
sys = mjcf.loads(mjcf_string)

# Set up initial state with the puck's initial position
init_q = jp.array([puck_pos[0], puck_pos[1], puck_pos[2], 1.0, 0.0, 0.0, 0.0])

# Target the first obstacle with initial velocity
first_obstacle = obstacles[0]
direction = jp.array([first_obstacle[0], first_obstacle[1]]) - jp.array([puck_pos[0], puck_pos[1]])
direction = direction / jp.sqrt(jp.sum(direction ** 2))  # Normalize
speed = 2.0  # Reduced initial speed so we can see acceleration effects better

# Add angular velocity to the puck (rotating around z-axis)
angular_speed = 0.0  # Start with no rotation

# Initial velocities: [vx, vy, vz, wx, wy, wz]
init_qd = jp.array([
    direction[0] * speed,  # vx
    direction[1] * speed,  # vy
    0.0,  # vz
    0.0,  # wx
    0.0,  # wy
    angular_speed  # wz - angular velocity around z-axis
])

# Choose the physics pipeline
from brax.spring import pipeline

# Initialize the physics state
state = pipeline.init(sys, init_q, init_qd)

# Ray sensor parameters
max_ray_distance = 3.0  # Maximum ray distance

# Define multiple ray angles (relative to puck orientation)
ray_angles = [0.0, 2 * jp.pi / 20, -2 * jp.pi / 20]

# Convert obstacles to JAX arrays for more efficient processing
obstacle_data = jp.array([[x, y, radius] for x, y, radius, _ in obstacles])


# ===== Functional ray distance sensing code =====

def ray_circle_intersection(ray_origin, ray_direction, circle_center, circle_radius):
    """
    Calculate intersection distance between ray and circle.
    Returns distance to intersection (inf if no intersection).

    Args:
        ray_origin: [x, y] position of ray origin
        ray_direction: Normalized [dx, dy] direction of ray
        circle_center: [x, y] position of circle center
        circle_radius: radius of circle

    Returns:
        distance: Distance to intersection or inf if no intersection
    """
    # Vector from ray origin to circle center
    to_center = circle_center - ray_origin

    # Project this vector onto ray direction
    proj_length = jp.dot(to_center, ray_direction)

    # If circle is behind the ray, no intersection
    no_intersection_behind = proj_length < 0

    # Find closest point on ray to circle center
    closest_point = ray_origin + proj_length * ray_direction

    # Distance from closest point to circle center
    dist_to_center = jp.linalg.norm(circle_center - closest_point)

    # If this distance > radius, no intersection
    no_intersection_distance = dist_to_center > circle_radius

    # Calculate distance from ray origin to intersection point
    # Using Pythagorean theorem
    to_intersection = jp.sqrt(jp.maximum(0.0, circle_radius ** 2 - dist_to_center ** 2))
    distance = proj_length - to_intersection

    # Return inf if no intersection, otherwise the distance
    return jp.where(
        no_intersection_behind | no_intersection_distance,
        jp.inf,
        jp.maximum(0.0, distance)  # Ensure distance is non-negative
    )


def ray_sensor_reading(ray_origin, ray_direction, obstacles, max_distance):
    """
    Calculate the distance from ray origin to the nearest obstacle.

    Args:
        ray_origin: [x, y] position of ray origin
        ray_direction: Normalized [dx, dy] direction of ray
        obstacles: Array of [x, y, radius] for each obstacle
        max_distance: Maximum ray distance to check

    Returns:
        min_distance: Distance to nearest intersection
        hit_point: [x, y] coordinate of intersection
        hit_object_idx: Index of hit obstacle, or -1 if none
    """
    # Initialize with maximum distance
    min_distance = max_distance
    hit_object_idx = -1

    # Function to check each obstacle
    def check_obstacle(i, min_data):
        min_dist, obj_idx = min_data

        # Get obstacle data
        obstacle = obstacles[i]
        center = obstacle[:2]
        radius = obstacle[2]

        # Calculate intersection distance
        dist = ray_circle_intersection(ray_origin, ray_direction, center, radius)

        # Update if closer
        closer = (dist < min_dist) & (dist >= 0.0001)  # Small epsilon to avoid self-intersection
        new_min_dist = jp.where(closer, dist, min_dist)
        new_obj_idx = jp.where(closer, i, obj_idx)

        return (new_min_dist, new_obj_idx)

    # Check all obstacles using a loop
    min_data = (min_distance, hit_object_idx)
    for i in range(len(obstacles)):
        min_data = check_obstacle(i, min_data)

    # After checking all obstacles
    min_distance, hit_object_idx = min_data

    # Calculate hit point
    hit_point = ray_origin + min_distance * ray_direction

    return min_distance, hit_point, hit_object_idx


def cast_rays(puck_pos, puck_radius, puck_angle, obstacles, max_distance, ray_angles):
    """
    Cast multiple rays from the puck at different angles and get distance readings.

    Args:
        puck_pos: [x, y] position of puck center
        puck_radius: Radius of the puck
        puck_angle: Current orientation angle of the puck
        obstacles: Array of obstacle data [x, y, radius]
        max_distance: Maximum ray distance
        ray_angles: List of angles (relative to puck orientation) for ray sensors

    Returns:
        results: List of (distance, hit_point, ray_dir, ray_origin) for each ray
    """
    results = []

    for angle_offset in ray_angles:
        # Calculate absolute ray angle
        ray_angle = puck_angle + angle_offset

        # Ray direction
        ray_dir = jp.array([jp.cos(ray_angle), jp.sin(ray_angle)])

        # Ray starts at edge of puck
        ray_origin = puck_pos + ray_dir * puck_radius

        # Cast ray
        distance, hit_point, hit_idx = ray_sensor_reading(
            ray_origin, ray_dir, obstacles, max_distance)

        # Store results
        results.append((distance, hit_point, ray_dir, ray_origin))

    return results


# Define control parameters
forward_thrust = 0.2  # Forward thrust force magnitude
rotation_torque = 1.  # Clockwise rotation torque magnitude


# Function to simulate with position tracking and ray sensing
def simulate(state, n_steps=100):
    states = [state]
    puck_positions = []
    all_ray_results = []  # List to store ray results for each frame

    # Apply control forces properly in the physics simulation
    def controlled_step(state):
        # Get current orientation
        quat = state.x.rot[0]
        puck_angle = 2 * jp.arctan2(quat[3], quat[0])

        # Calculate direction vector for thrust
        forward_dir = jp.array([jp.cos(puck_angle), jp.sin(puck_angle), 0.0])

        # Create a force vector for thrust in the forward direction
        # This is in world coordinates
        thrust_force = forward_dir * forward_thrust

        # Apply rotation torque around z-axis (negative for clockwise)
        # This is in world coordinates
        rotation = jp.array([0.0, 0.0, -rotation_torque])

        # Compute center of mass position and velocity in world frame
        x_i = state.x_i
        xd_i = state.xd_i

        # Create force/torque to apply at center of mass
        # Force will accelerate the body in the forward direction
        # Torque will rotate the body clockwise
        force = Force(
            vel=thrust_force,  # Linear force
            ang=rotation  # Angular torque
        )

        # Apply the force to create a delta-velocity
        # F = ma, so dv = F/m * dt
        # Assuming unit mass for simplicity
        dv = Motion(
            vel=force.vel * sys.opt.timestep,
            ang=force.ang * sys.opt.timestep
        )

        # Add the delta-velocity to current velocity
        new_xd_i = Motion(
            vel=xd_i.vel + dv.vel,
            ang=xd_i.ang + dv.ang
        )

        # Take a physics step with the standard pipeline
        next_state = pipeline.step(sys, state, None)

        # Replace the velocity with our controlled version
        # This simulates applying the force to the body
        controlled_state = next_state.replace(
            xd_i=new_xd_i
        )

        # Enforce 2D constraint
        controlled_state = controlled_state.replace(
            x=controlled_state.x.replace(
                pos=controlled_state.x.pos.at[:, 2].set(0.0)
            ),
            xd=controlled_state.xd.replace(
                vel=controlled_state.xd.vel.at[:, 2].set(0.0),
                ang=controlled_state.xd.ang.at[:, :2].set(0.0)  # Keep z-axis rotation
            )
        )

        return controlled_state

    # JIT-compile our step function for performance
    jitted_step = jax.jit(controlled_step)

    for i in range(n_steps):
        # Get current position and orientation
        puck_pos = state.x.pos[0, :2]
        quat = state.x.rot[0]
        puck_angle = 2 * jp.arctan2(quat[3], quat[0])

        # Cast multiple rays
        ray_results = cast_rays(
            puck_pos,
            puck_radius,
            puck_angle,
            obstacle_data,
            max_ray_distance,
            ray_angles
        )

        # Record data
        puck_positions.append(puck_pos)
        all_ray_results.append(ray_results)

        # Take a controlled step
        state = jitted_step(state)
        states.append(state)

    # Add final position and ray information
    puck_pos = state.x.pos[0, :2]
    puck_positions.append(puck_pos)

    # Cast rays for final state
    quat = state.x.rot[0]
    puck_angle = 2 * jp.arctan2(quat[3], quat[0])
    ray_results = cast_rays(
        puck_pos,
        puck_radius,
        puck_angle,
        obstacle_data,
        max_ray_distance,
        ray_angles
    )
    all_ray_results.append(ray_results)

    return states, puck_positions, all_ray_results


# Run the simulation
states, puck_positions, all_ray_results = simulate(state, 500)

# Define line styles for different rays - using same color for all rays
ray_line_styles = ['-', '--', ':']  # Different line styles for each ray

# Create an animation of the simulation
fig, ax = plt.subplots(figsize=(10, 8))

# Set up the plot
ax.set_xlim(-box_size / 2 - puck_radius, box_size / 2 + puck_radius)
ax.set_ylim(-box_size / 2 - puck_radius, box_size / 2 + puck_radius)
ax.set_aspect('equal')
ax.set_title('2D Puck Simulation with Controls and Ray Sensors')
ax.grid(True)

# Draw the box
box = Rectangle((-box_size / 2, -box_size / 2), box_size, box_size,
                facecolor='none', edgecolor='black', linewidth=2)
ax.add_patch(box)

# Draw the obstacles
obstacle_patches = []
for i, (x, y, radius, color_str) in enumerate(obstacles):
    color_values = [float(c) for c in color_str.split()[:3]]  # Extract RGB from RGBA string
    obstacle = Circle((x, y), radius, facecolor=color_values, edgecolor='black')
    ax.add_patch(obstacle)
    obstacle_patches.append(obstacle)

# Initialize the puck with initial position from state
puck = Circle((puck_positions[0][0], puck_positions[0][1]),
              puck_radius, facecolor='purple', edgecolor='black')
ax.add_patch(puck)

# Initialize ray sensor visualization (now three rays)
ray_lines = []
ray_ends = []

for i, style in enumerate(ray_line_styles):
    line, = ax.plot([], [], 'y' + style, lw=2)  # Yellow ray with different line styles
    end, = ax.plot([], [], 'yo', ms=6)  # Yellow endpoint
    ray_lines.append(line)
    ray_ends.append(end)

# Initialize velocity vector line
vel_line, = ax.plot([], [], 'r-', lw=2)
arrowhead, = ax.plot([], [], 'r.', ms=10)

# Initialize acceleration vector line (shows applied force)
acc_line, = ax.plot([], [], 'b-', lw=2)  # Blue line for acceleration
acc_arrowhead, = ax.plot([], [], 'b.', ms=10)  # Blue dot for acceleration arrowhead

# Initialize rotation indicator line (to show angular velocity)
rotation_line, = ax.plot([], [], 'g-', lw=3)  # Green line from center to edge

# Add trail for the puck
puck_trail, = ax.plot([], [], '-', color='blue', alpha=0.5, linewidth=1)

# Add text for frame count and ray distances
frame_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top')
distance_texts = []
for i in range(3):  # Create text for each ray
    text = ax.text(0.05, 0.90 - i * 0.05, '', transform=ax.transAxes, verticalalignment='top')
    distance_texts.append(text)
angular_vel_text = ax.text(0.05, 0.75, '', transform=ax.transAxes, verticalalignment='top')
controls_text = ax.text(0.05, 0.70, '', transform=ax.transAxes, verticalalignment='top')

# Add a legend for ray styles
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='y', linestyle=ray_line_styles[0], lw=2, label='Forward Ray'),
    Line2D([0], [0], color='y', linestyle=ray_line_styles[1], lw=2, label='Left Ray (+120°)'),
    Line2D([0], [0], color='y', linestyle=ray_line_styles[2], lw=2, label='Right Ray (-120°)'),
    Line2D([0], [0], color='r', lw=2, label='Velocity'),
    Line2D([0], [0], color='b', lw=2, label='Thrust Force'),
    Line2D([0], [0], color='g', lw=2, label='Orientation')
]
ax.legend(handles=legend_elements, loc='upper right')


# Animation update function
def update(frame, trail_length=20):
    if frame >= len(states):
        frame = len(states) - 1

    # Get the current state
    state = states[frame]

    # Update puck position
    if frame < len(puck_positions):
        puck_x, puck_y = puck_positions[frame]
    else:
        puck_x, puck_y = state.x.pos[0, 0], state.x.pos[0, 1]

    puck.center = (puck_x, puck_y)

    # Update velocity vector
    vx, vy = state.xd.vel[0, 0], state.xd.vel[0, 1]
    scale = 0.2
    vel_line.set_data([puck_x, puck_x + vx * scale], [puck_y, puck_y + vy * scale])
    arrowhead.set_data([puck_x + vx * scale], [puck_y + vy * scale])

    # Get the puck angle
    quat = state.x.rot[0]
    angle = 2 * np.arctan2(quat[3], quat[0])

    # Calculate thrust force direction
    dir_x, dir_y = np.cos(angle), np.sin(angle)

    # Update thrust vector (blue)
    acc_scale = 0.3
    acc_line.set_data(
        [puck_x, puck_x + dir_x * forward_thrust * acc_scale],
        [puck_y, puck_y + dir_y * forward_thrust * acc_scale]
    )
    acc_arrowhead.set_data(
        [puck_x + dir_x * forward_thrust * acc_scale],
        [puck_y + dir_y * forward_thrust * acc_scale]
    )

    # Update all ray sensor visualizations
    if frame < len(all_ray_results):
        ray_results = all_ray_results[frame]

        for i, ((ray_dist, ray_hit, ray_dir, ray_origin), line, end, text) in enumerate(
                zip(ray_results, ray_lines, ray_ends, distance_texts)):

            # Update ray visualization
            if ray_dist < max_ray_distance:
                line.set_data(
                    [ray_origin[0], ray_hit[0]],
                    [ray_origin[1], ray_hit[1]]
                )
                end.set_data([ray_hit[0]], [ray_hit[1]])
            else:
                # Use max distance if no hit
                ray_end_pt = ray_origin + ray_dir * max_ray_distance
                line.set_data(
                    [ray_origin[0], ray_end_pt[0]],
                    [ray_origin[1], ray_end_pt[1]]
                )
                end.set_data([ray_end_pt[0]], [ray_end_pt[1]])

            # Update distance text
            if i == 0:
                text.set_text(f'Forward Ray: {ray_dist:.2f}')
            elif i == 1:
                text.set_text(f'Left Ray: {ray_dist:.2f}')
            else:
                text.set_text(f'Right Ray: {ray_dist:.2f}')

    # Update rotation indicator line using quaternion orientation
    if frame < len(states):
        # Calculate the point on the edge of the puck
        edge_x = puck_x + puck_radius * np.cos(angle)
        edge_y = puck_y + puck_radius * np.sin(angle)

        # Update the rotation indicator line
        rotation_line.set_data([puck_x, edge_x], [puck_y, edge_y])

    # Update trail with limited positions
    trail_start = max(0, frame + 1 - trail_length)
    trail_end = frame + 1

    # Update trail with all positions up to current frame
    puck_trail.set_data([p[0] for p in puck_positions[trail_start:trail_end]],
                        [p[1] for p in puck_positions[trail_start:trail_end]])

    # Update frame text
    frame_text.set_text(f'Frame: {frame}')

    # Add angular velocity text
    if frame < len(states):
        wz = state.xd.ang[0, 2]
        angular_vel_text.set_text(f'Angular Velocity: {wz:.2f} rad/s')

    # Add controls text
    controls_text.set_text(f'Controls: Forward Thrust={forward_thrust:.1f}, Rotation Torque={rotation_torque:.1f}')

    return [puck, vel_line, arrowhead, acc_line, acc_arrowhead, puck_trail, rotation_line,
            frame_text, angular_vel_text, controls_text] + ray_lines + ray_ends + distance_texts


# Create the animation
animation = FuncAnimation(fig, update, frames=len(states),
                          interval=100, blit=True)

# Display the animation
plt.show()