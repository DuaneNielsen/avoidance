import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import numpy as np

# Import Brax
from brax.io import mjcf

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

# Begin building the MJCF string
mjcf_string = """
<mujoco model="2d_puck">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 0" />
  <custom>
    <numeric data="0.5" name="elasticity"/>
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
speed = 5.0

# Add angular velocity to the puck (rotating around z-axis)
angular_speed = 2.0  # radians per second

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


def cast_ray(puck_pos, puck_radius, obstacles, max_distance, sensor_angle):
    """
    Cast a ray from the puck at a fixed angle and get distance reading.

    Args:
        puck_pos: [x, y] position of puck center
        puck_radius: Radius of the puck
        obstacles: Array of obstacle data [x, y, radius]
        max_distance: Maximum ray distance
        sensor_angle: Fixed sensor angle

    Returns:
        distance: Distance to nearest obstacle or max_distance
        hit_point: Position of hit
        ray_dir: Direction of ray
        ray_origin: Origin of ray
    """
    # Fixed ray direction
    ray_dir = jp.array([jp.cos(sensor_angle), jp.sin(sensor_angle)])

    # Ray starts at edge of puck
    ray_origin = puck_pos + ray_dir * puck_radius

    # Cast ray
    distance, hit_point, hit_idx = ray_sensor_reading(
        ray_origin, ray_dir, obstacles, max_distance)

    return distance, hit_point, ray_dir, ray_origin


# Function to simulate with position tracking and ray sensing
def simulate(state, n_steps=100):
    states = [state]
    puck_positions = []
    ray_readings = []  # Store ray sensor readings
    ray_directions = []  # Store ray directions
    ray_hit_points = []  # Store ray hit points
    ray_origins = []  # Store ray origin points

    # JIT-compile the step function
    step_fn = jax.jit(pipeline.step)

    for i in range(n_steps):
        # Get current position and orientation
        puck_pos = state.x.pos[0, :2]

        # Extract the quaternion (w, x, y, z) from the state
        quat = state.x.rot[0]  # Quaternion for the puck

        # Convert quaternion to rotation angle around z-axis
        angle = 2 * jp.arctan2(quat[3], quat[0])

        # Calculate ray direction based on puck's orientation
        ray_dir = jp.array([jp.cos(angle), jp.sin(angle)])

        # Ray starts at edge of puck in the direction of orientation
        ray_origin = puck_pos + ray_dir * puck_radius

        # Cast ray
        distance, hit_point, hit_idx = ray_sensor_reading(
            ray_origin, ray_dir, obstacle_data, max_ray_distance)

        # Record data
        puck_positions.append(puck_pos)
        ray_readings.append(distance)
        ray_directions.append(ray_dir)
        ray_hit_points.append(hit_point)
        ray_origins.append(ray_origin)

        # Take a step
        state = step_fn(sys, state, None)

        # Enforce 2D constraint
        state = state.replace(
            x=state.x.replace(
                pos=state.x.pos.at[:, 2].set(0.0)
            ),
            xd=state.xd.replace(
                vel=state.xd.vel.at[:, 2].set(0.0),
                ang=state.xd.ang.at[:].set(0.0)
            )
        )

        states.append(state)

    # Add final position and ray information
    puck_pos = state.x.pos[0, :2]
    puck_positions.append(puck_pos)

    # Cast ray for final state
    quat = state.x.rot[0]
    angle = 2 * jp.arctan2(quat[3], quat[0])
    ray_dir = jp.array([jp.cos(angle), jp.sin(angle)])
    ray_origin = puck_pos + ray_dir * puck_radius
    distance, hit_point, hit_idx = ray_sensor_reading(
        ray_origin, ray_dir, obstacle_data, max_ray_distance)

    ray_readings.append(distance)
    ray_directions.append(ray_dir)
    ray_hit_points.append(hit_point)
    ray_origins.append(ray_origin)

    return states, puck_positions, ray_readings, ray_directions, ray_hit_points, ray_origins


# Run the simulation
states, puck_positions, ray_readings, ray_directions, ray_hit_points, ray_origins = simulate(state, 500)

# Create an animation of the simulation
fig, ax = plt.subplots(figsize=(10, 8))

# Set up the plot
ax.set_xlim(-box_size / 2 - puck_radius, box_size / 2 + puck_radius)
ax.set_ylim(-box_size / 2 - puck_radius, box_size / 2 + puck_radius)
ax.set_aspect('equal')
ax.set_title('2D Puck Simulation with Rotating Ray Sensor')
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

# Initialize ray sensor visualization
ray_line, = ax.plot([], [], 'y-', lw=2)  # Yellow ray
ray_end, = ax.plot([], [], 'yo', ms=6)  # Yellow endpoint

# Initialize velocity vector line
vel_line, = ax.plot([], [], 'r-', lw=2)
arrowhead, = ax.plot([], [], 'r.', ms=10)

# Initialize rotation indicator line (to show angular velocity)
rotation_line, = ax.plot([], [], 'g-', lw=3)  # Green line from center to edge

# Add trail for the puck
puck_trail, = ax.plot([], [], '-', color='blue', alpha=0.5, linewidth=1)

# Add text for frame count, distance reading, and angular velocity
frame_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top')
distance_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, verticalalignment='top')
angular_vel_text = ax.text(0.05, 0.85, '', transform=ax.transAxes, verticalalignment='top')


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

    # Update ray sensor visualization
    if frame < len(ray_directions):
        ray_origin = ray_origins[frame]
        ray_hit = ray_hit_points[frame]
        ray_dist = ray_readings[frame]

        if ray_dist < max_ray_distance:
            ray_line.set_data(
                [ray_origin[0], ray_hit[0]],
                [ray_origin[1], ray_hit[1]]
            )
            ray_end.set_data([ray_hit[0]], [ray_hit[1]])
        else:
            # Use max distance if no hit
            ray_dir = ray_directions[frame]
            ray_end_pt = ray_origin + ray_dir * max_ray_distance
            ray_line.set_data(
                [ray_origin[0], ray_end_pt[0]],
                [ray_origin[1], ray_end_pt[1]]
            )
            ray_end.set_data([ray_end_pt[0]], [ray_end_pt[1]])

    # Update rotation indicator line using quaternion orientation
    if frame < len(states):
        # Extract the quaternion (w, x, y, z) from the state
        quat = state.x.rot[0]  # Quaternion for the puck

        # We need to convert the quaternion to a rotation angle in 2D
        # In a 2D case, we care about rotation around z-axis
        # Quaternion to rotation angle around z-axis can be found through:
        # 2 * atan2(z, w)
        angle = 2 * np.arctan2(quat[3], quat[0])

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

    # Update text information
    frame_text.set_text(f'Frame: {frame}')
    if frame < len(ray_readings):
        distance_text.set_text(f'Ray Distance: {ray_readings[frame]:.2f}')

    # Add angular velocity text
    if frame < len(states):
        wz = state.xd.ang[0, 2]
        angular_vel_text.set_text(f'Angular Velocity: {wz:.2f} rad/s')

    return [puck, vel_line, arrowhead, puck_trail, ray_line, ray_end, rotation_line,
            frame_text, distance_text, angular_vel_text]


# Create the animation
animation = FuncAnimation(fig, update, frames=len(states),
                          interval=100, blit=True)

# Display the animation
plt.show()