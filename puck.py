# Import necessary libraries
import jax
from jax import numpy as jp
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    from IPython.display import HTML
except ImportError:
    pass  # Not using IPython

import brax
from brax.io import mjcf
from brax.positional import pipeline


def create_environment():
    """Create a simple environment with just a controllable puck in XY plane"""
    # Create MJCF XML for the environment - just the puck
    xml = """<mujoco>
      <option timestep="0.01" gravity="0 0 0"/>
      <default>
        <geom contype="1" conaffinity="1" friction="0.0 0.0 0.0"/>
      </default>
      <worldbody>
        <!-- Controllable puck -->
        <body name="puck" pos="0 0 0.1">
          <joint name="free" type="free" damping="0.0"/>
          <geom name="puck_geom" size="0.3" type="sphere" rgba="0.2 0.6 0.8 1" mass="0.1"/>
        </body>
        <!-- No floor - just open space -->
      </worldbody>
    </mujoco>
    """

    # Load the environment
    env = mjcf.loads(xml)

    return env


def apply_thrust(state, thrust_x, thrust_y):
    """Apply thrust forces to the puck"""
    # Create force vector (x, y, z)
    force = jp.array([thrust_x, thrust_y, 0.0])

    # Get current position and velocity for logging
    pos = state.x.pos[0]
    vel = state.xd.vel[0]

    # Apply force by directly setting the velocity (more reliable than adding)
    # For simplicity, just use the thrust direction and magnitude directly
    new_vel = jp.array([thrust_x, thrust_y, 0.0])

    # Apply velocity to the puck (assuming it's body index 0)
    new_state = state.replace(xd=state.xd.replace(vel=state.xd.vel.at[0].set(new_vel)))

    # For debugging - print when applying significant thrust
    if abs(thrust_x) > 0.5 or abs(thrust_y) > 0.5:
        print(f"  Applied thrust: ({thrust_x:.1f}, {thrust_y:.1f})")
        print(f"  Before: Pos=({float(pos[0]):.2f}, {float(pos[1]):.2f}), "
              f"Vel=({float(vel[0]):.2f}, {float(vel[1]):.2f})")
        print(f"  Set velocity to: {new_vel}")

    return new_state


def run_simulation(num_steps=500, render_every=5):
    """Run the simulation manually by directly updating position"""
    states = []
    curr_state = globals()['state']  # Get the current state from globals

    # Create position and velocity arrays for manual movement
    position = jp.array([0.0, 0.0, 0.1])
    velocity = jp.array([0.0, 0.0, 0.0])

    # Store initial state
    states.append(curr_state)

    # Display initial position
    print(f"Initial position: {position}")

    # Simulation loop
    for i in range(1, num_steps):
        # Apply thrust based on pattern
        thrust_x = 0.0
        thrust_y = 0.0

        # Simple automated thrust pattern to demonstrate movement
        if i % 100 < 25:
            thrust_x = 0.1  # Right
        elif i % 100 < 50:
            thrust_y = 0.1  # Up
        elif i % 100 < 75:
            thrust_x = -0.1  # Left
        else:
            thrust_y = -0.1  # Down

        # Update velocity based on thrust
        velocity = jp.array([velocity[0] + thrust_x, velocity[1] + thrust_y, 0.0])

        # Add damping to prevent infinite acceleration
        velocity = velocity * 0.98

        # Update position based on velocity
        position = jp.array([
            position[0] + velocity[0],
            position[1] + velocity[1],
            position[2]
        ])

        # Trace position updates every 10 steps
        if i % 10 == 0:
            print(f"Step {i}: Position = ({float(position[0]):.2f}, {float(position[1]):.2f}), "
                  f"Velocity = ({float(velocity[0]):.2f}, {float(velocity[1]):.2f})")

        # Create a new state with the updated position and velocity
        if i % render_every == 0:
            # Create copies of the current state with updated position and velocity
            new_pos = curr_state.x.pos.at[0].set(position)
            new_vel = curr_state.xd.vel.at[0].set(velocity)

            new_state = curr_state.replace(
                x=curr_state.x.replace(pos=new_pos),
                xd=curr_state.xd.replace(vel=new_vel)
            )

            states.append(new_state)

    # Return the states for visualization
    return states


def visualize_simulation(states, env):
    """Create visualization of the puck's movement"""
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.grid(True)
    plt.title('Controllable Puck Simulation (XY Plane)')

    # Create patch for the puck
    puck = Circle(xy=(0, 0), radius=0.3, color='blue', alpha=0.7)
    ax.add_patch(puck)

    # Create lines for showing trail
    trail_lines = []
    for _ in range(20):  # Maximum trail length
        line = Line2D([0, 0], [0, 0], color='blue', alpha=0, linewidth=2)
        ax.add_line(line)
        trail_lines.append(line)

    # Create patches for thrust indicators
    thrust_indicators = []
    for color in ['green', 'green', 'green', 'green']:
        rect = Rectangle((0, 0), 0.05, 0.05, color=color, alpha=0)
        ax.add_patch(rect)
        thrust_indicators.append(rect)

    # Store the last 20 positions for trail effect
    trail_positions = []

    def animate(frame):
        # Get current frame state
        if frame >= len(states):
            return [puck] + trail_lines + thrust_indicators

        # Update puck position
        pos = states[frame].x.pos[0]
        puck.center = (float(pos[0]), float(pos[1]))

        # Print debug info
        if frame % 10 == 0:
            print(f"Visualizing frame {frame}: Position = ({float(pos[0]):.2f}, {float(pos[1]):.2f})")

        # Update trail
        trail_positions.append((float(pos[0]), float(pos[1])))
        if len(trail_positions) > 20:
            trail_positions.pop(0)

        # Update trail lines
        for i in range(len(trail_lines)):
            line = trail_lines[i]
            if i < len(trail_positions) - 1:
                line.set_data(
                    [trail_positions[i][0], trail_positions[i + 1][0]],
                    [trail_positions[i][1], trail_positions[i + 1][1]]
                )
                line.set_alpha(i / len(trail_positions))
            else:
                line.set_alpha(0)

        # Get velocity
        vel = states[frame].xd.vel[0]
        x_vel = float(vel[0])
        y_vel = float(vel[1])

        # Update thrust indicators
        thrust_n, thrust_e, thrust_s, thrust_w = thrust_indicators

        # North thruster (y+)
        if y_vel > 0.05:
            thrust_n.set_alpha(min(1.0, y_vel))
            thrust_n.set_xy([float(pos[0]) - 0.025, float(pos[1])])
            thrust_n.set_width(0.05)
            thrust_n.set_height(0.3)
        else:
            thrust_n.set_alpha(0)

        # East thruster (x+)
        if x_vel > 0.05:
            thrust_e.set_alpha(min(1.0, x_vel))
            thrust_e.set_xy([float(pos[0]) - 0.3, float(pos[1]) - 0.025])
            thrust_e.set_width(0.3)
            thrust_e.set_height(0.05)
        else:
            thrust_e.set_alpha(0)

        # South thruster (y-)
        if y_vel < -0.05:
            thrust_s.set_alpha(min(1.0, -y_vel))
            thrust_s.set_xy([float(pos[0]) - 0.025, float(pos[1]) - 0.3])
            thrust_s.set_width(0.05)
            thrust_s.set_height(0.3)
        else:
            thrust_s.set_alpha(0)

        # West thruster (x-)
        if x_vel < -0.05:
            thrust_w.set_alpha(min(1.0, -x_vel))
            thrust_w.set_xy([float(pos[0]), float(pos[1]) - 0.025])
            thrust_w.set_width(0.3)
            thrust_w.set_height(0.05)
        else:
            thrust_w.set_alpha(0)

        return [puck] + trail_lines + thrust_indicators

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(states), interval=50, blit=True
    )

    # Save animation to file or display it
    plt.tight_layout()

    # Try various output options with error handling
    try:
        anim.save('puck_simulation.mp4', writer='ffmpeg', fps=20)
        return "Animation saved as 'puck_simulation.mp4'"
    except Exception as e:
        print(f"Could not save as MP4: {e}")

        try:
            anim.save('puck_simulation.gif', writer='pillow', fps=15)
            return "Animation saved as 'puck_simulation.gif'"
        except Exception as e2:
            print(f"Could not save as GIF: {e2}")

            plt.show()
            return "Animation displayed directly"


# Main execution block
if __name__ == "__main__":
    print("Starting puck simulation...")

    # Create the environment
    env = create_environment()
    print("Environment created")

    # Set elasticity for better bouncing off walls (if any)
    env = env.replace(elasticity=jp.array([0.9] * env.ngeom))

    # Initialize the simulation
    qd = jp.zeros(6)  # Initial velocity (zero)
    state = jax.jit(pipeline.init)(env, env.init_q, qd)
    print("Simulation initialized")

    # Print initial position
    pos = state.x.pos[0]
    print(f"Initial puck position: ({float(pos[0]):.2f}, {float(pos[1]):.2f}, {float(pos[2]):.2f})")

    # Store the state for global access
    globals()['state'] = state
    globals()['env'] = env

    # Run the simulation
    print("\nRunning simulation with position tracing:")
    print("----------------------------------------")
    simulation_states = run_simulation(num_steps=800, render_every=4)
    print("----------------------------------------")
    print(f"Simulation complete with {len(simulation_states)} states")

    # Print a few positions to verify movement
    for i, state in enumerate(simulation_states[:5]):
        pos = state.x.pos[0]
        print(f"State {i}: Position = ({float(pos[0]):.2f}, {float(pos[1]):.2f}, {float(pos[2]):.2f})")

    # Visualize the results
    print("\nCreating visualization...")
    result = visualize_simulation(simulation_states, env)
    print(result)