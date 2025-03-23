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
init_qd = jp.array([direction[0] * speed, direction[1] * speed, 0.0, 0.0, 0.0, 0.0])

# Choose the physics pipeline
from brax.spring import pipeline

# Initialize the physics state
state = pipeline.init(sys, init_q, init_qd)

# Ray sensor parameters
sensor_angle = 0  # Initial angle in radians (0 = right direction)
max_ray_distance = 3.0  # Maximum ray distance


# Function to calculate ray-circle intersection distance
def ray_distance_to_obstacle(ray_origin, ray_direction, obstacles, walls, max_distance):
    """
    Calculate the distance from ray origin to the nearest obstacle intersection.

    Args:
        ray_origin: [x, y] position of ray origin
        ray_direction: Normalized [dx, dy] direction of ray
        obstacles: List of [x, y, radius] for each obstacle
        walls: List of walls as [x1, y1, x2, y2]
        max_distance: Maximum ray distance to check

    Returns:
        min_distance: Distance to nearest intersection or max_distance if none found
        hit_point: [x, y] coordinate of intersection or None
        hit_object: Index of hit obstacle or wall, or None
    """
    min_distance = max_distance
    hit_point = None
    hit_object = None

    # Check intersections with obstacles (circles)
    for i, (x, y, radius, _) in enumerate(obstacles):
        # Vector from ray origin to obstacle center
        to_center = jp.array([x, y]) - ray_origin

        # Project this vector onto the ray direction
        proj_length = jp.dot(to_center, ray_direction)

        # If the obstacle is behind the ray, skip it
        if proj_length < 0:
            continue

        # Find the closest point on the ray to the obstacle center
        closest_point = ray_origin + proj_length * ray_direction

        # Calculate the distance from this point to the obstacle center
        dist_to_center = jp.linalg.norm(jp.array([x, y]) - closest_point)

        # If this distance is greater than the radius, no intersection
        if dist_to_center > radius:
            continue

        # Calculate the distance from the ray origin to the intersection point
        # Using Pythagorean theorem
        to_intersection = jp.sqrt(radius ** 2 - dist_to_center ** 2)
        distance = proj_length - to_intersection

        # If this is closer than our current minimum, update
        if 0 <= distance < min_distance:
            min_distance = distance
            hit_point = ray_origin + distance * ray_direction
            hit_object = f"obstacle_{i}"

    # Check intersections with walls (box edges)
    # Left wall
    if ray_direction[0] < 0:  # Only check if ray is pointing left
        t = (-box_size / 2 - ray_origin[0]) / ray_direction[0]
        if 0 <= t < min_distance:
            y_intersect = ray_origin[1] + t * ray_direction[1]
            if -box_size / 2 <= y_intersect <= box_size / 2:
                min_distance = t
                hit_point = jp.array([-box_size / 2, y_intersect])
                hit_object = "left_wall"

    # Right wall
    if ray_direction[0] > 0:  # Only check if ray is pointing right
        t = (box_size / 2 - ray_origin[0]) / ray_direction[0]
        if 0 <= t < min_distance:
            y_intersect = ray_origin[1] + t * ray_direction[1]
            if -box_size / 2 <= y_intersect <= box_size / 2:
                min_distance = t
                hit_point = jp.array([box_size / 2, y_intersect])
                hit_object = "right_wall"

    # Bottom wall
    if ray_direction[1] < 0:  # Only check if ray is pointing down
        t = (-box_size / 2 - ray_origin[1]) / ray_direction[1]
        if 0 <= t < min_distance:
            x_intersect = ray_origin[0] + t * ray_direction[0]
            if -box_size / 2 <= x_intersect <= box_size / 2:
                min_distance = t
                hit_point = jp.array([x_intersect, -box_size / 2])
                hit_object = "bottom_wall2"

    # Top wall
    if ray_direction[1] > 0:  # Only check if ray is pointing up
        t = (box_size / 2 - ray_origin[1]) / ray_direction[1]
        if 0 <= t < min_distance:
            x_intersect = ray_origin[0] + t * ray_direction[0]
            if -box_size / 2 <= x_intersect <= box_size / 2:
                min_distance = t
                hit_point = jp.array([x_intersect, box_size / 2])
                hit_object = "top_wall"

    return min_distance, hit_point, hit_object


# Function to simulate with position tracking and ray sensing
def simulate(state, n_steps=100):
    states = [state]
    puck_positions = []
    ray_readings = []  # Store ray sensor readings
    ray_directions = []  # Store ray directions
    ray_hits = []  # Store ray hit points

    # Get obstacle data for ray casting
    obstacle_data = [[x, y, radius] for x, y, radius, _ in obstacles]
    walls = []  # We'll check walls separately

    # JIT-compile the step function
    step_fn = jax.jit(pipeline.step)

    for i in range(n_steps):
        # Get current position and velocity
        puck_pos = state.x.pos[0, :2]
        puck_vel = state.xd.vel[0, :2]

        # Determine ray direction based on velocity if moving
        vel_mag = jp.linalg.norm(puck_vel)
        if vel_mag > 0.1:
            # Use velocity direction for ray
            ray_dir = puck_vel / vel_mag
        else:
            # Use fixed angle if very slow
            ray_dir = jp.array([jp.cos(sensor_angle), jp.sin(sensor_angle)])

        # Cast ray from the edge of the puck in the ray direction
        ray_origin = puck_pos + ray_dir * puck_radius

        # Get distance to nearest obstacle
        distance, hit_point, hit_object = ray_distance_to_obstacle(
            ray_origin, ray_dir, obstacles, walls, max_ray_distance)

        # Record puck position and ray reading
        puck_positions.append(puck_pos.tolist())
        ray_readings.append(distance)
        ray_directions.append(ray_dir)
        ray_hits.append(hit_point)

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

    # Add final position
    puck_positions.append(state.x.pos[0, :2].tolist())

    # Get final ray reading
    puck_pos = state.x.pos[0, :2]
    puck_vel = state.xd.vel[0, :2]
    vel_mag = jp.linalg.norm(puck_vel)
    if vel_mag > 0.1:
        ray_dir = puck_vel / vel_mag
    else:
        ray_dir = jp.array([jp.cos(sensor_angle), jp.sin(sensor_angle)])
    ray_origin = puck_pos + ray_dir * puck_radius
    distance, hit_point, hit_object = ray_distance_to_obstacle(
        ray_origin, ray_dir, obstacles, walls, max_ray_distance)
    ray_readings.append(distance)
    ray_directions.append(ray_dir)
    ray_hits.append(hit_point)

    return states, puck_positions, ray_readings, ray_directions, ray_hits


# Run the simulation
states, puck_positions, ray_readings, ray_directions, ray_hits = simulate(state, 500)

# Create an animation of the simulation
fig, ax = plt.subplots(figsize=(10, 8))

# Set up the plot
ax.set_xlim(-box_size / 2 - puck_radius, box_size / 2 + puck_radius)
ax.set_ylim(-box_size / 2 - puck_radius, box_size / 2 + puck_radius)
ax.set_aspect('equal')
ax.set_title('2D Puck Simulation with Ray Distance Sensor')
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

# Add trail for the puck
puck_trail, = ax.plot([], [], '-', color='blue', alpha=0.5, linewidth=1)

# Add text for frame count and distance reading
frame_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top')
distance_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, verticalalignment='top')


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
        ray_dir = ray_directions[frame]
        ray_dist = ray_readings[frame]
        ray_origin = np.array([puck_x, puck_y]) + ray_dir * puck_radius

        # If we have a hit point, use it directly
        if ray_hits[frame] is not None:
            ray_end_pt = ray_hits[frame]
            ray_line.set_data(
                [ray_origin[0], ray_end_pt[0]],
                [ray_origin[1], ray_end_pt[1]]
            )
            ray_end.set_data([ray_end_pt[0]], [ray_end_pt[1]])
        else:
            # Otherwise use the max distance
            ray_end_pt = ray_origin + ray_dir * ray_dist
            ray_line.set_data(
                [ray_origin[0], ray_end_pt[0]],
                [ray_origin[1], ray_end_pt[1]]
            )
            ray_end.set_data([ray_end_pt[0]], [ray_end_pt[1]])

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

    return [puck, vel_line, arrowhead, puck_trail, ray_line, ray_end, frame_text, distance_text]


# Create the animation
animation = FuncAnimation(fig, update, frames=len(states),
                          interval=100, blit=True)

# Display the animation
plt.show()