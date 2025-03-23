import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import numpy as np

# Import Brax
from brax import base
from brax.io import mjcf

# Create an MJCF model for our 2D puck in a box with a static obstacle
box_size = 2.0
puck_radius = 0.2
obstacle_radius = 0.3

# Make sure obstacle and puck are at DIFFERENT positions but closer to each other
obstacle_pos = [0.2, 0.3, 0]  # Position of the static obstacle
puck_pos = [-0.2, 0.1, 0]  # Position of the puck - closer to obstacle

# MJCF model with a static obstacle (as a geom instead of a body)
# Let's try using a cylinder with minimal height - just at the threshold
mjcf_string = f"""
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
    <body name="puck" pos="{puck_pos[0]} {puck_pos[1]} {puck_pos[2]}">
      <joint type="free" name="puck_joint"/>
      <geom name="puck_geom" type="sphere" size="{puck_radius}" mass="1" rgba="0.7 0.2 0.8 1"/>
    </body>

    <!-- Static obstacle - using a cylinder with minimal height -->
    <geom name="obstacle" type="cylinder" pos="{obstacle_pos[0]} {obstacle_pos[1]} {obstacle_pos[2]}" 
          size="{obstacle_radius} 0.0001" rgba="0.2 0.7 0.2 1" contype="1" conaffinity="1"/>

    <!-- Walls of the box -->
    <geom name="bottom_wall" type="plane" pos="0 0 -0.1" size="{box_size} {box_size} 0.1" rgba="0.8 0.8 0.8 1" conaffinity="1"/>
    <geom name="left_wall" type="plane" pos="-{box_size / 2} 0 0" size="0.1 {box_size} {box_size}" rgba="0.8 0.8 0.8 1" conaffinity="1" zaxis="1 0 0"/>
    <geom name="right_wall" type="plane" pos="{box_size / 2} 0 0" size="0.1 {box_size} {box_size}" rgba="0.8 0.8 0.8 1" conaffinity="1" zaxis="-1 0 0"/>
    <geom name="top_wall" type="plane" pos="0 {box_size / 2} 0" size="{box_size} 0.1 {box_size}" rgba="0.8 0.8 0.8 1" conaffinity="1" zaxis="0 -1 0"/>
    <geom name="bottom_wall2" type="plane" pos="0 -{box_size / 2} 0" size="{box_size} 0.1 {box_size}" rgba="0.8 0.8 0.8 1" conaffinity="1" zaxis="0 1 0"/>
  </worldbody>
</mujoco>
"""

# Load the MJCF model into Brax
sys = mjcf.loads(mjcf_string)

# Set up initial state with the puck's initial position
# Don't use the default initialization, set the position explicitly
init_q = jp.array([puck_pos[0], puck_pos[1], puck_pos[2], 1.0, 0.0, 0.0, 0.0])

# Velocity toward the obstacle
direction = jp.array([obstacle_pos[0], obstacle_pos[1]]) - jp.array([puck_pos[0], puck_pos[1]])
direction = direction / jp.sqrt(jp.sum(direction ** 2))  # Normalize
speed = 5.0
init_qd = jp.array([direction[0] * speed, direction[1] * speed, 0.0, 0.0, 0.0, 0.0])

# Choose the physics pipeline
from brax.spring import pipeline

# Initialize the physics state
state = pipeline.init(sys, init_q, init_qd)

# Check initial separation - now using geometry info directly
puck_radius = 0.2  # Use the same value as defined above
obstacle_radius = 0.3  # Use the same value as defined above
# Use the obstacle_pos defined earlier
obstacle_pos = jp.array(obstacle_pos)

# Get puck position from state
puck_pos = state.x.pos[0]

initial_distance = jp.sqrt(jp.sum((puck_pos[:2] - obstacle_pos[:2]) ** 2))
min_required_distance = puck_radius + obstacle_radius
print(f"Initial separation: {initial_distance}")
print(f"Minimum required separation: {min_required_distance}")
print(f"Gap between surfaces: {initial_distance - min_required_distance}")

if initial_distance < min_required_distance:
    print("ERROR: Puck and obstacle are overlapping at start!")
else:
    print("OK: Puck and obstacle properly separated at start")


# Function to print system geometries
def print_system_geometries(system):
    """Print information about all geometries in a Brax system."""
    for i in range(system.ngeom):
        print(f'geometry_{i}: {sys.geom_pos[i]}')


# Function to simulate with position tracking and detailed debug info
def simulate(state, n_steps=100):
    states = [state]

    # Track puck positions
    puck_positions = []
    detailed_debug = []

    # JIT-compile the step function
    step_fn = jax.jit(pipeline.step)

    # For debugging, dump the full system info first
    print("\nSystem state at start:")
    print(f"System has {sys.nbody} bodies")
    print(f"System has {sys.ngeom} geoms")

    # Print detailed geometry information
    print_system_geometries(sys)

    # Print link information
    print("\nLink information:")
    if hasattr(state, 'x'):
        print(f"Number of links in state: {len(state.x.pos)}")
        for i in range(len(state.x.pos)):
            print(f"Link {i} position: {state.x.pos[i]}")

    # Determine obstacle position using geometry information
    obstacle_pos_in_sys = jp.array(obstacle_pos)  # Default position

    # Look for the obstacle in the geometry information
    if hasattr(sys, 'geom') and hasattr(sys.geom, 'pos'):
        # In this setup, geom 1 should be the obstacle (index starts at 0)
        # Geom 0 would be the puck's geometry
        obstacle_idx = 1
        if obstacle_idx < len(sys.geom.pos):
            obstacle_pos_in_sys = sys.geom.pos[obstacle_idx]
            print(f"\nUsing obstacle at position {obstacle_pos_in_sys} from system geometry")

    print(f"Using obstacle position: {obstacle_pos_in_sys}")

    for i in range(n_steps):
        # if i == 165:
        #     breakpoint()
        # Record puck position before step
        puck_pos = state.x.pos[0].copy()
        puck_positions.append(puck_pos[:2].tolist())

        # Calculate distance using obstacle position extracted from system
        distance = jp.sqrt(jp.sum((state.x.pos[0, :2] - obstacle_pos_in_sys[:2]) ** 2))

        # Print debug info at regular intervals
        if i % 20 == 0 or (i > 0 and i < 10):
            print(f"Step {i}:")
            print(f"  Puck position: {state.x.pos[0]}")
            print(f"  Puck velocity: {state.xd.vel[0]}")
            print(f"  Distance to obstacle: {distance}")
            print(f"  Min required distance: {min_required_distance}")

            # Check for contacts in this state
            if hasattr(state, 'contact') and state.contact is not None:
                print(f"  Contact detected!")
                if hasattr(state.contact, 'pos'):
                    print(f"  Contact position: {state.contact.pos}")

            print_system_geometries(sys)

        debug_entry = {
            'step': i,
            'puck_pos': state.x.pos[0].copy(),
            'puck_vel': state.xd.vel[0].copy(),
            'distance': float(distance),
            'penetration': float(max(0, min_required_distance - distance))
        }
        detailed_debug.append(debug_entry)

        # Take a step
        prev_state = state
        state = step_fn(sys, state, None)

        # Check for sudden position changes (possible penetration fixes)
        pos_change = jp.linalg.norm(state.x.pos[0] - prev_state.x.pos[0])
        vel_mag = jp.linalg.norm(prev_state.xd.vel[0])
        expected_change = vel_mag * sys.opt.timestep

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

    # Print summary of potential penetrations
    penetrations = [d['penetration'] for d in detailed_debug]
    if max(penetrations) > 0:
        worst_step = penetrations.index(max(penetrations))
        print(f"\nWORST PENETRATION at step {worst_step}:")
        print(f"  Penetration depth: {max(penetrations)}")
        print(f"  Puck position: {detailed_debug[worst_step]['puck_pos']}")
        print(f"  Puck velocity: {detailed_debug[worst_step]['puck_vel']}")

    return states, puck_positions, detailed_debug


# Run the simulation
states, puck_positions, debug_info = simulate(state, 500)

# Print the system configuration for debugging
print("\nSystem information:")
print(f"Number of bodies: {sys.nbody}")
print(f"Number of geoms: {sys.ngeom}")
print(f"Timestep: {sys.opt.timestep}")
print(f"Elasticity: {sys.elasticity if hasattr(sys, 'elasticity') else 'Not directly accessible'}")

for step in debug_info:
    print(step)

# Create an animation of the simulation
fig, ax = plt.subplots(figsize=(8, 8))

# Set up the plot
ax.set_xlim(-box_size / 2 - puck_radius, box_size / 2 + puck_radius)
ax.set_ylim(-box_size / 2 - puck_radius, box_size / 2 + puck_radius)
ax.set_aspect('equal')
ax.set_title('2D Puck Simulation with Static Obstacle')
ax.grid(True)

# Draw the box
box = Rectangle((-box_size / 2, -box_size / 2), box_size, box_size,
                facecolor='none', edgecolor='black', linewidth=2)
ax.add_patch(box)

# Draw the static obstacle - now static, doesn't move with animation
obstacle = Circle((obstacle_pos[0], obstacle_pos[1]),
                  obstacle_radius, facecolor='green', edgecolor='black')
ax.add_patch(obstacle)

# Initialize the puck with initial position from state
puck = Circle((puck_positions[0][0], puck_positions[0][1]),
              puck_radius, facecolor='purple', edgecolor='black')
ax.add_patch(puck)

# Initialize velocity vector line
vel_line, = ax.plot([], [], 'r-', lw=2)
arrowhead, = ax.plot([], [], 'r.', ms=10)

# Add trail for the puck
puck_trail, = ax.plot([], [], '-', color='blue', alpha=0.5, linewidth=1)

# Add text for positions and simulation info
puck_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top')
distance_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, verticalalignment='top')
frame_text = ax.text(0.05, 0.85, '', transform=ax.transAxes, verticalalignment='top')


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

    # Update trail with limited positions
    trail_start = max(0, frame + 1 - trail_length)
    trail_end = frame + 1

    # Update trail with all positions up to current frame
    puck_trail.set_data([p[0] for p in puck_positions[trail_start:trail_end]],
                        [p[1] for p in puck_positions[trail_start:trail_end]])

    # Calculate distance for current frame
    dist = jp.sqrt((puck_x - obstacle_pos[0]) ** 2 + (puck_y - obstacle_pos[1]) ** 2)

    # Update text
    puck_text.set_text(f'Puck: ({puck_x:.4f}, {puck_y:.4f})')
    distance_text.set_text(f'Distance: {dist:.4f} (Min req: {min_required_distance:.4f})')
    frame_text.set_text(f'Frame: {frame}')

    return [puck, vel_line, arrowhead, puck_trail, puck_text, distance_text, frame_text]


# Create the animation
animation = FuncAnimation(fig, update, frames=len(states),
                          interval=100, blit=True)

# Display the animation
plt.show()