import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

# Import Brax
from brax.io import mjcf

# Create an MJCF model for our 2D puck in a box with a static obstacle
box_size = 2.0
puck_radius = 0.2
obstacle_radius = 0.3

# Make sure obstacle and puck are at different positions
obstacle_pos = [0.2, 0.3, 0]  # Position of the static obstacle
puck_pos = [-0.2, 0.1, 0]  # Position of the puck

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


# Function to simulate with position tracking
def simulate(state, n_steps=100):
    states = [state]
    puck_positions = []

    # JIT-compile the step function
    step_fn = jax.jit(pipeline.step)

    for i in range(n_steps):
        # Record puck position
        puck_positions.append(state.x.pos[0, :2].tolist())

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

    return states, puck_positions


# Run the simulation
states, puck_positions = simulate(state, 500)

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

# Draw the static obstacle
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

# Add text for frame count
frame_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top')


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

    # Update frame text
    frame_text.set_text(f'Frame: {frame}')

    return [puck, vel_line, arrowhead, puck_trail, frame_text]


# Create the animation
animation = FuncAnimation(fig, update, frames=len(states),
                          interval=100, blit=True)

# Display the animation
plt.show()