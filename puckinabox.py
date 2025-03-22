import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import numpy as np

# Import Brax
from brax import base
from brax.io import mjcf

# Create an MJCF model for our 2D puck in a box
box_size = 2.0  # 2x2 meter box
puck_radius = 0.2

# Define the MJCF model as a string
mjcf_string = f"""
<mujoco model="2d_puck">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>

  <!-- Disable gravity, we want a pure 2D physics simulation -->
  <option timestep="0.01" gravity="0 0 0" />

  <custom>
    <!-- Parameters for Brax physics -->
    <numeric data="0.99" name="elasticity"/>  <!-- Slightly less than perfect elastic collisions -->
    <numeric data="0.0" name="ang_damping"/>  <!-- No angular damping -->
    <numeric data="0.0" name="vel_damping"/>  <!-- No velocity damping -->
  </custom>

  <worldbody>
    <!-- The puck: a sphere constrained to the xy-plane -->
    <body name="puck" pos="0 0 0">
      <!-- Free joint but we'll zero out z movement in our code -->
      <joint type="free" name="puck_joint"/>
      <geom name="puck_geom" type="sphere" size="{puck_radius}" mass="1" rgba="0.7 0.2 0.8 1"/>
    </body>

    <!-- Walls of the box -->
    <!-- Bottom wall -->
    <geom name="bottom_wall" type="plane" pos="0 0 -0.1" size="{box_size} {box_size} 0.1" rgba="0.8 0.8 0.8 1" conaffinity="1"/>
    <!-- Left wall -->
    <geom name="left_wall" type="plane" pos="-{box_size / 2} 0 0" size="0.1 {box_size} {box_size}" rgba="0.8 0.8 0.8 1" conaffinity="1" zaxis="1 0 0"/>
    <!-- Right wall -->
    <geom name="right_wall" type="plane" pos="{box_size / 2} 0 0" size="0.1 {box_size} {box_size}" rgba="0.8 0.8 0.8 1" conaffinity="1" zaxis="-1 0 0"/>
    <!-- Top wall -->
    <geom name="top_wall" type="plane" pos="0 {box_size / 2} 0" size="{box_size} 0.1 {box_size}" rgba="0.8 0.8 0.8 1" conaffinity="1" zaxis="0 -1 0"/>
    <!-- Bottom wall -->
    <geom name="bottom_wall2" type="plane" pos="0 -{box_size / 2} 0" size="{box_size} 0.1 {box_size}" rgba="0.8 0.8 0.8 1" conaffinity="1" zaxis="0 1 0"/>
  </worldbody>
</mujoco>
"""

# Load the MJCF model into Brax
sys = mjcf.loads(mjcf_string)

# Create initial conditions: position in middle, random velocity
# Use a JAX random key for reproducibility
key = jax.random.PRNGKey(42)  # You can change the seed for different initial conditions
key, subkey = jax.random.split(key)

# Initial position: center of the box
init_q = jp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # [x, y, z, qw, qx, qy, qz]

# Initial velocity: random in x and y, zero in other dimensions
# Sample from normal distribution, scale to get reasonable speed
vx, vy = jax.random.normal(subkey, (2,)) * 1.0  # Scale factor for velocity
init_qd = jp.array([vx, vy, 0.0, 0.0, 0.0, 0.0])  # [vx, vy, vz, wx, wy, wz]

# Choose the physics pipeline
from brax.positional import pipeline

# Initialize the physics state
state = pipeline.init(sys, init_q, init_qd)


# Function to simulate for n steps
def simulate(state, n_steps=100):
    states = [state]

    # JIT-compile the step function for faster execution
    step_fn = jax.jit(pipeline.step)

    for _ in range(n_steps):
        # Take a step with no control inputs (None)
        state = step_fn(sys, state, None)

        # Enforce 2D constraint: zero out z position and all z velocities
        # This keeps the puck perfectly in the xy-plane
        state = state.replace(
            x=state.x.replace(
                pos=state.x.pos.at[:, 2].set(0.0)
            ),
            xd=state.xd.replace(
                vel=state.xd.vel.at[:, 2].set(0.0),
                ang=state.xd.ang.at[:].set(0.0)  # Zero out all angular velocities
            )
        )

        states.append(state)

    return states


# Run the simulation for 500 steps
states = simulate(state, 500)

# Create an animation of the simulation
fig, ax = plt.subplots(figsize=(8, 8))

# Set up the plot
ax.set_xlim(-box_size / 2 - puck_radius, box_size / 2 + puck_radius)
ax.set_ylim(-box_size / 2 - puck_radius, box_size / 2 + puck_radius)
ax.set_aspect('equal')
ax.set_title('2D Puck Simulation')
ax.grid(True)

# Draw the box
box = Rectangle((-box_size / 2, -box_size / 2), box_size, box_size,
                facecolor='none', edgecolor='black', linewidth=2)
ax.add_patch(box)

# Initialize the puck
puck = Circle((0, 0), puck_radius, facecolor='purple', edgecolor='black')
ax.add_patch(puck)

# Initialize velocity vector line (instead of arrow for easier updating)
vel_line, = ax.plot([], [], 'r-', lw=2)
arrowhead, = ax.plot([], [], 'r.', ms=10)  # Simple dot for arrowhead

# Add a trail (will show the last 20 positions)
trail_length = 20
trail_line, = ax.plot([], [], 'o-', color='blue', alpha=0.5, markersize=2)
trail_data = {'x': [], 'y': []}

# Add text for position and velocity
position_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top')
velocity_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, verticalalignment='top')


# Animation update function
def update(frame):
    # Get the current state
    state = states[frame]

    # Update puck position
    x, y = state.x.pos[0, 0], state.x.pos[0, 1]
    puck.center = (x, y)

    # Update velocity vector
    vx, vy = state.xd.vel[0, 0], state.xd.vel[0, 1]
    speed = float(jp.sqrt(vx ** 2 + vy ** 2))

    # Scale velocity for visualization
    scale = 0.2
    vel_line.set_data([x, x + vx * scale], [y, y + vy * scale])
    arrowhead.set_data([x + vx * scale], [y + vy * scale])

    # Update trail
    trail_data['x'].append(x)
    trail_data['y'].append(y)
    if len(trail_data['x']) > trail_length:
        trail_data['x'] = trail_data['x'][-trail_length:]
        trail_data['y'] = trail_data['y'][-trail_length:]
    trail_line.set_data(trail_data['x'], trail_data['y'])

    # Update text
    position_text.set_text(f'Position: ({x:.2f}, {y:.2f})')
    velocity_text.set_text(f'Velocity: ({vx:.2f}, {vy:.2f}) - Speed: {speed:.2f}')

    return [puck, vel_line, arrowhead, trail_line, position_text, velocity_text]


# Create the animation
animation = FuncAnimation(fig, update, frames=len(states),
                          interval=20, blit=True)

# Display the animation in a window
plt.show()

# Uncomment to save as a file instead:
# animation.save('puck_simulation.mp4', writer='ffmpeg', fps=30, dpi=100)
# animation.save('puck_simulation.gif', writer='pillow', fps=30)