import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
import time

# For visualization of one instance
import mujoco.viewer as viewer

# Install with: pip install noise
try:
    from noise import pnoise2

    HAS_NOISE = True
except ImportError:
    print("Install noise library with: pip install noise")
    HAS_NOISE = False

xml_template = """
<mujoco>
    <asset>
        <hfield name="terrain" nrow="64" ncol="64" size="8 8 3 0.2"/>
        <texture name="grid" type="2d" builtin="checker" 
                 width="512" height="512" rgb2="0 0 0" rgb1="1 1 1"/>
        <material name="grid" texture="grid" texrepeat="8 8" 
                  texuniform="true" reflectance="0.3"/>
    </asset>
    <worldbody>
        <light name="top" pos="0 0 5"/>
        <geom name="ground" type="hfield" hfield="terrain" material="grid"/>
        <body name="robot" pos="0 0 4">
            <freejoint/>
            <geom name="body" type="capsule" size="0.3 0.8" rgba="0 0 1 1"/>
        </body>
    </worldbody>
</mujoco>
"""


def generate_perlin_terrain(nrow, ncol, scale=0.1, octaves=4, seed=None):
    """Generate Perlin noise terrain"""
    if not HAS_NOISE:
        np.random.seed(seed)
        return np.random.uniform(0, 1, nrow * ncol)

    terrain = np.zeros((nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            terrain[i][j] = pnoise2(
                i * scale, j * scale,
                octaves=octaves, persistence=0.5,
                lacunarity=2.0, base=seed or 0
            )

    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    return terrain.flatten()


def generate_hills_terrain(nrow, ncol, num_hills=5, seed=None):
    """Generate terrain with Gaussian hills"""
    np.random.seed(seed)
    terrain = np.zeros((nrow, ncol))

    for _ in range(num_hills):
        center_x = np.random.randint(5, nrow - 5)
        center_y = np.random.randint(5, ncol - 5)
        sigma_x = np.random.uniform(3, 8)
        sigma_y = np.random.uniform(3, 8)
        height = np.random.uniform(0.5, 1.0)

        for i in range(nrow):
            for j in range(ncol):
                dist_sq = ((i - center_x) / sigma_x) ** 2 + ((j - center_y) / sigma_y) ** 2
                terrain[i, j] += height * np.exp(-dist_sq)

    terrain = np.clip(terrain, 0, 1)
    return terrain.flatten()


def generate_waves_terrain(nrow, ncol, seed=None):
    """Generate wave-like terrain"""
    np.random.seed(seed)
    freq_x = np.random.uniform(0.5, 2.0)
    freq_y = np.random.uniform(0.5, 2.0)
    phase_x = np.random.uniform(0, 2 * np.pi)
    phase_y = np.random.uniform(0, 2 * np.pi)

    x = np.linspace(0, 4 * np.pi * freq_x, nrow)
    y = np.linspace(0, 4 * np.pi * freq_y, ncol)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (np.sin(X + phase_x) + np.cos(Y + phase_y))
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    return Z.flatten()


def generate_random_terrain(nrow, ncol, seed=None):
    """Generate random terrain with smoothing"""
    np.random.seed(seed)
    terrain = np.random.uniform(0, 1, (nrow, ncol))

    # Apply smoothing
    for _ in range(2):
        terrain = 0.25 * (np.roll(terrain, 1, axis=0) +
                          np.roll(terrain, -1, axis=0) +
                          np.roll(terrain, 1, axis=1) +
                          np.roll(terrain, -1, axis=1))
    return terrain.flatten()


class MultiInstanceTerrainGenerator:
    def __init__(self, num_instances):
        self.num_instances = num_instances
        self.terrain_types = ['perlin', 'hills', 'random', 'waves']
        self.terrain_generators = {
            'perlin': generate_perlin_terrain,
            'hills': generate_hills_terrain,
            'random': generate_random_terrain,
            'waves': generate_waves_terrain
        }

    def generate_batch(self, nrow, ncol, seeds, terrain_types=None):
        """Generate terrain for multiple instances"""
        if terrain_types is None:
            # Assign different terrain types to different instances
            terrain_types = [self.terrain_types[i % len(self.terrain_types)]
                             for i in range(self.num_instances)]

        batch_terrain = []
        for i in range(self.num_instances):
            terrain_type = terrain_types[i]
            seed = seeds[i]

            generator = self.terrain_generators[terrain_type]
            if terrain_type == 'hills':
                terrain = generator(nrow, ncol, num_hills=np.random.randint(3, 8), seed=seed)
            else:
                terrain = generator(nrow, ncol, seed=seed)

            batch_terrain.append(terrain)

        return np.array(batch_terrain)


def create_mjx_models_with_terrain(num_instances, nrow=64, ncol=64):
    """Create multiple MJX models with different terrain"""

    # Generate different terrains for each instance
    terrain_gen = MultiInstanceTerrainGenerator(num_instances)
    seeds = np.random.randint(0, 10000, num_instances)
    terrain_batch = terrain_gen.generate_batch(nrow, ncol, seeds)

    models = []
    datas = []

    for i in range(num_instances):
        # Create model with specific terrain
        mj_model = mujoco.MjModel.from_xml_string(xml_template)
        mj_model.hfield_data[:] = terrain_batch[i]

        # Convert to MJX
        mjx_model = mjx.put_model(mj_model)
        mjx_data = mjx.put_data(mj_model, mujoco.MjData(mj_model))

        models.append(mjx_model)
        datas.append(mjx_data)

    return models, datas, terrain_batch, seeds


@jax.jit
def step_batch(models, datas):
    """Vectorized step function for multiple instances"""

    def single_step(model, data):
        return mjx.step(model, data)

    # Use vmap to vectorize across instances
    return jax.vmap(single_step)(models, datas)


@jax.jit
def reset_batch(models, datas, keys):
    """Reset all instances with different random keys"""

    def single_reset(model, data, key):
        # Reset to initial state with some randomization
        data = data.replace(
            qpos=data.qpos.at[:3].set(jax.random.uniform(key, (3,), minval=-0.5, maxval=0.5)),
            qvel=data.qvel.at[:].set(0)
        )
        return mjx.forward(model, data)

    return jax.vmap(single_reset)(models, datas, keys)


def visualize_instance(models, datas, instance_idx=0):
    """Visualize one specific instance using regular MuJoCo viewer"""
    # Convert back to regular MuJoCo for visualization
    mj_model = models[instance_idx]
    mj_data = mjx.get_data(mj_model, datas[instance_idx])

    # Need to convert mjx model back to regular mujoco model
    # This is a bit tricky, so we'll create a new model
    mj_model_vis = mujoco.MjModel.from_xml_string(xml_template)

    # Copy the terrain data from the mjx model
    terrain_data = np.array(mj_model.hfield_data)
    mj_model_vis.hfield_data[:] = terrain_data

    return mj_model_vis, mj_data


def main():
    print("MJX Multi-Instance Dynamic Terrain Demo")

    # Configuration
    num_instances = 8
    nrow, ncol = 64, 64

    print(f"Creating {num_instances} parallel simulations...")

    # Create models and initial data
    models, datas, terrain_info, seeds = create_mjx_models_with_terrain(num_instances, nrow, ncol)

    # Convert to JAX arrays for batched operations
    models = jax.tree_map(lambda *xs: jnp.stack(xs), *models)
    datas = jax.tree_map(lambda *xs: jnp.stack(xs), *datas)

    print(f"Generated terrains with seeds: {seeds}")

    # Initialize random keys for resets
    key = jax.random.PRNGKey(42)

    # Simulation parameters
    steps_per_terrain = 1000
    terrain_changes = 5
    total_steps = 0

    # Optional: Visualize the first instance
    vis_model, vis_data = visualize_instance([models[0]], [datas[0]], 0)

    print("\nStarting parallel simulations...")
    print("Close viewer window to continue with batch simulation only")

    # Try to show visualization of first instance
    try:
        with viewer.launch_passive(vis_model, vis_data) as v:
            step_count = 0
            while v.is_running() and step_count < 200:  # Short visualization
                # Step the batch simulation
                datas = step_batch(models, datas)

                # Update visualization with first instance
                vis_data_updated = mjx.get_data(vis_model, datas[0])
                vis_data.qpos[:] = vis_data_updated.qpos
                vis_data.qvel[:] = vis_data_updated.qvel

                mujoco.mj_forward(vis_model, vis_data)
                v.sync()
                step_count += 1
                total_steps += 1
    except:
        print("Visualization not available, continuing with batch simulation...")

    print(f"\nRunning headless batch simulation for {num_instances} instances...")

    # Main simulation loop (headless)
    for terrain_change in range(terrain_changes):
        print(f"\nTerrain change {terrain_change + 1}/{terrain_changes}")

        # Generate new terrains
        new_seeds = np.random.randint(0, 10000, num_instances)
        terrain_gen = MultiInstanceTerrainGenerator(num_instances)
        new_terrain_batch = terrain_gen.generate_batch(nrow, ncol, new_seeds)

        # Update models with new terrain (this would require recreating models in practice)
        # For demonstration, we'll just reset positions
        reset_keys = jax.random.split(key, num_instances)
        datas = reset_batch(models, datas, reset_keys)
        key = jax.random.split(key)[0]

        # Run simulation steps
        step_times = []
        for step in range(steps_per_terrain):
            start_time = time.time()

            # Step all instances in parallel
            datas = step_batch(models, datas)

            step_time = time.time() - start_time
            step_times.append(step_time)
            total_steps += num_instances

            # Print progress occasionally
            if step % 200 == 0:
                avg_step_time = np.mean(step_times[-100:]) if len(step_times) >= 100 else np.mean(step_times)
                sps = num_instances / avg_step_time  # Steps per second
                print(f"  Step {step}/{steps_per_terrain}, "
                      f"Avg time: {avg_step_time * 1000:.2f}ms, "
                      f"SPS: {sps:.0f}")

        avg_step_time = np.mean(step_times)
        total_sps = num_instances / avg_step_time
        print(f"  Completed terrain {terrain_change + 1}: "
              f"Avg {avg_step_time * 1000:.2f}ms/step, "
              f"{total_sps:.0f} steps/sec")

    print(f"\nSimulation completed!")
    print(f"Total steps simulated: {total_steps}")
    print(f"Final robot positions (z-coordinate):")

    # Show final positions
    for i in range(min(num_instances, 8)):  # Show first 8 instances
        z_pos = float(datas.qpos[i, 2])  # Z position of robot
        print(f"  Instance {i + 1}: z = {z_pos:.3f}")


if __name__ == "__main__":
    main()