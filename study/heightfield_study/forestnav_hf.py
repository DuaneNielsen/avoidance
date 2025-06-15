import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx
from dataclasses import dataclass
from mujoco.mjx._src.math import normalize_with_norm

import mediapy as media
from math import sin, cos
from tqdm import trange
from typing import Callable, List
from functools import partial


def generate_flat_terrain(nrow=32, ncol=32, height=0.0, seed=None):
    """Generate flat terrain with uniform height"""
    terrain = np.full((nrow, ncol), height)
    return terrain.flatten()


def generate_hills_terrain(nrow=32, ncol=32, seed=42, height=1.0):
    """Generate hills terrain using the algorithm from the first script"""
    np.random.seed(seed)
    terrain = np.zeros((nrow, ncol))

    # Generate 3-6 hills
    for _ in range(np.random.randint(3, 6)):
        cx = np.random.randint(5, nrow - 5)
        cy = np.random.randint(5, ncol - 5)
        sigma = np.random.uniform(3, 6)
        height = np.random.uniform(low=-0.02, high=height)

        for x in range(nrow):
            for y in range(ncol):
                dist_sq = ((x - cx) / sigma) ** 2 + ((y - cy) / sigma) ** 2
                terrain[x, y] += height * np.exp(-dist_sq)

    terrain = np.clip(terrain, 0, 1)
    return terrain.flatten()

def generate_column_terrain(nrow=32, ncol=32, column_size=3, gap_size=2, column_height=1.0):
    """Generate terrain with square columns and gaps, with clear center area"""
    terrain = np.zeros((nrow, ncol))

    # Calculate center area bounds (4x4 clear area)
    center_start_row = nrow // 2 - 2
    center_end_row = nrow // 2 + 2
    center_start_col = ncol // 2 - 2
    center_end_col = ncol // 2 + 2

    # Create grid pattern of columns
    for row in range(0, nrow, column_size + gap_size):
        for col in range(0, ncol, column_size + gap_size):
            # Define column bounds
            row_end = min(row + column_size, nrow)
            col_end = min(col + column_size, ncol)

            # Check if this column overlaps with center area
            if (row < center_end_row and row_end > center_start_row and
                    col < center_end_col and col_end > center_start_col):
                # Skip columns that overlap with center area
                continue

            # Create the column
            terrain[row:row_end, col:col_end] = column_height

    return terrain.flatten()


def generate_obstacles(verts, radius=0.07):
    xml = ""

    for i, (x, y) in enumerate(verts.T):
        xml += f"""
        <!-- Obstacle {i} -->
        <body pos="{x} {y} 0" name="obstacle_{i}">
          <joint name="obstacle_{i}_x_joint" type="slide" axis="1. 0. 0."/>
          <joint name="obstacle_{i}_y_joint" type="slide" axis="0. 1. 0."/>
          <geom name="obstacle_geom_{i}" type="sphere" pos="0. 0. 0" size="{radius}" contype="1" conaffinity="1" material="obstacle_material" solimp="0.9 0.95 0.001" solref="0.004 1"/>
        </body>
    """
    return xml


def forestnav_xml(
        goal_coord,
        sensor_angle=0.6,
        num_sensors=128,
        obstacles_xml_f: Callable = None
):
    """

    :param goal_coord:
    :param sensor_angle:
    :param num_sensors:
    :param obstacles_xml_f:
    :return:
    """

    xml = """
    <mujoco model="forestnav_v1">
      <compiler autolimits="true"/>

      <option integrator="implicitfast"/>

      <asset>
        <material name="body_material" rgba="0.5 0.5 0.9 1"/>
        <material name="obstacle_material" rgba="0.8 0.2 0.2 1"/>
        <material name="goal_material" rgba="0.3 0.9 0.2 1"/>
        <material name="terrain_material" rgba="0.7 0.5 0.3 1"/>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.4 0.6 0.2" rgb2="0.6 0.8 0.4"/>
        <material name="grid_material" texture="grid" texrepeat="4 4" reflectance="0.3"/>
        <hfield name="terrain" nrow="32" ncol="32" size="4 4 6.0 0.1"/>
      </asset>

      <option timestep="0.02">
        <!--flag contact="disable" /-->
      </option>

      <default>
        <joint damping="1.0" stiffness="0."/>
      </default>

      <worldbody>
    """

    xml += f"""
        <site name="origin"/>
        <light pos="0 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <light pos="3 3 3" dir="-1 -1 -1" diffuse="0.4 0.4 0.4"/>

        <!-- Add the terrain heightfield -->
        <geom name="ground" type="hfield" hfield="terrain" material="grid_material"/>

        <!-- stacked joint: hinge + slide -->
        <body pos="0.0 0 0.2" name="vehicle">
          <joint name="x_joint" type="slide" axis="1. 0. 0." range="-3 3"/>
          <joint name="y_joint" type="slide" axis="0. 1. 0." range="-3 3"/>
          <joint name="rot_joint" type="hinge" axis="0 0 1."/>
          <site name="velocity_site" pos="0 0 0" size="0.01"/>
          <frame pos="0 0.01 0" quat="-1 1 0 0">
          <geom name="vehicle_body" type="sphere" pos="0 0 0" size="0.015" mass="0.5" material="body_material"/>
          <site name="vehicle_collision_site" type="sphere" pos="0 0 0" size="0.016" mass="0." material="body_material" contype="1" conaffinity="1" 
                solimp="0.9 0.95 0.001" solref="0.004 1"/>
          """

    rangefinder_angles = np.linspace(start=-sensor_angle, stop=sensor_angle, num=num_sensors)
    for i, theta in enumerate(rangefinder_angles):
        xml += f"""
                  <site name="site_rangefinder{i}" quat="{cos(theta / 2)} 0 {sin(theta / 2)} 0" size="0.01" rgba="1 0 0 1"/>
                """

    xml += f"""
          </frame>
        </body>

        <!-- goal -->
        <geom name="goal" pos="{goal_coord[0]} {goal_coord[1]} 2.0" type="sphere" size="0.07" material="goal_material" />
    """

    # xml += obstacles_xml_f()

    xml += """    
      </worldbody>

      <sensor>
      <framepos name="vehicle_pos" objtype="body" objname="vehicle"/>
      <framepos name="goal_pos" objtype="geom" objname="goal"/>
      <framepos name="goalvec" objtype="geom" objname="goal" reftype="site" refname="velocity_site"/>
      <touch name="vehicle_collision" site="vehicle_collision_site"/>
      """

    for i in range(num_sensors):
        xml += f"""
            <rangefinder name="rangefinder{i}" site="site_rangefinder{i}"/>
            """

    xml += """
      </sensor>

      <actuator>

        <!-- Forward/backward velocity control in body frame -->
        <velocity name="body_y" site='velocity_site' kv="1." gear="0 1 0 0 0 0" ctrlrange="-2 2"/>

        <!-- Angular velocity control around Z axis in body frame -->
        <velocity name="angular_velocity" joint="rot_joint" kv="1." ctrlrange="-1 1"/>
      </actuator>

    </mujoco>
    """

    return xml


import re
from typing import List, Union


def get_joint_indices(model, pattern: Union[str, List[str]], use_qvel: bool = False):
    """
    Get JAX index tensor for joints matching a pattern.

    Args:
        model: MuJoCo model
        pattern: String pattern (regex) or list of joint names
        use_qvel: If True, return qvel indices, otherwise qpos indices

    Returns:
        JAX array of indices that can be used to slice qpos/qvel
    """
    if isinstance(pattern, str):
        # Use regex pattern matching
        joint_names = []
        for i in range(model.njnt):
            joint_name = model.joint(i).name
            if re.search(pattern, joint_name):
                joint_names.append(joint_name)
    else:
        # Direct list of joint names
        joint_names = pattern

    # Get the appropriate indices
    if use_qvel:
        indices = [model.joint(name).dofadr[0] for name in joint_names]
    else:
        indices = [model.joint(name).qposadr[0] for name in joint_names]

    return jnp.array(indices, dtype=jnp.int32)


class JointIndexer:
    def __init__(self, model):
        self.model = model
        self._cache = {}

    def __call__(self, pattern, use_qvel=False):
        """Get indices with caching for performance"""
        # Convert list to tuple for hashing
        if isinstance(pattern, list):
            cache_key = (tuple(pattern), use_qvel)
        else:
            cache_key = (pattern, use_qvel)

        if cache_key not in self._cache:
            self._cache[cache_key] = get_joint_indices(self.model, pattern, use_qvel)
        return self._cache[cache_key]

    @property
    def vehicle(self):
        return self(['x_joint', 'y_joint', 'rot_joint'])

    @property
    def vehicle_vel(self):
        return self(['x_joint', 'y_joint', 'rot_joint'], use_qvel=True)

    @property
    def obstacles(self):
        return self(r'obstacle_\d+_[xy]_joint')

    def obstacle(self, obstacle_id):
        return self(f'obstacle_{obstacle_id}_[xy]_joint')


def create_coordinate_tensor(x_range, y_range):
    """
    Create a 2×N tensor where the first dimension contains x,y coordinates.

    Parameters:
    x_range: array-like, x coordinate values
    y_range: array-like, y coordinate values

    Returns:
    numpy.ndarray: shape (2, N) where N = len(x_range) * len(y_range)
                   First row contains x coordinates, second row contains y coordinates
    """
    # Create meshgrid
    X, Y = np.meshgrid(x_range, y_range)

    # Flatten and stack to create (2, N) tensor
    x_coords = X.flatten()
    y_coords = Y.flatten()

    # Stack to create 2×N tensor
    coord_tensor = np.stack([x_coords, y_coords], axis=0)

    return coord_tensor


def remove_vertices_inside_circles(vertices, circles):
    """
    Remove vertices that are inside any of the given circles.

    Parameters:
    -----------
    vertices : numpy.ndarray
        Shape (2, N) where first row contains x coordinates,
        second row contains y coordinates
    circles : numpy.ndarray
        Shape (3, M) where first row contains x centers,
        second row contains y centers, third row contains radii

    Returns:
    --------
    numpy.ndarray
        Shape (2, K) where K <= N, containing vertices outside all circles
    """
    # Extract coordinates
    x_vertices = vertices[0, :]  # Shape: (N,)
    y_vertices = vertices[1, :]  # Shape: (N,)

    # Extract circle parameters
    x_centers = circles[0, :]  # Shape: (M,)
    y_centers = circles[1, :]  # Shape: (M,)
    radii = circles[2, :]  # Shape: (M,)

    # Compute distances from each vertex to each circle center
    # Using broadcasting: (N, 1) - (1, M) = (N, M)
    dx = x_vertices[:, np.newaxis] - x_centers[np.newaxis, :]
    dy = y_vertices[:, np.newaxis] - y_centers[np.newaxis, :]
    distances_squared = dx ** 2 + dy ** 2

    # Check if each vertex is inside each circle
    # distances_squared has shape (N, M), radii has shape (M,)
    inside_circles = distances_squared <= (radii[np.newaxis, :]) ** 2

    # A vertex should be removed if it's inside ANY circle
    # any(axis=1) checks across circles for each vertex
    vertices_to_remove = np.any(inside_circles, axis=1)

    # Keep only vertices that are NOT inside any circle
    vertices_to_keep = ~vertices_to_remove

    return vertices[:, vertices_to_keep]


def uniform_like(key, x, minval=0.0, maxval=1.0):
    return jax.random.uniform(key, shape=x.shape, minval=minval, maxval=maxval)


def make_xml(sensor_angle, num_sensors):
    GOAL_POS = [4., 0.]
    VEHICLE_START_POS = [0., 0.]

    vertices = create_coordinate_tensor(
        x_range=np.linspace(-1, 5, 10),
        y_range=np.linspace(-1, 1, 5)
    )
    clearings = np.array([
        [VEHICLE_START_POS[0], GOAL_POS[0]],
        [VEHICLE_START_POS[1], GOAL_POS[1]],
        [0.1, 0.1]
    ])
    vertices = remove_vertices_inside_circles(vertices, clearings)

    obstacles_gen_f = partial(generate_obstacles, vertices, 0.07)

    xml = forestnav_xml(GOAL_POS, sensor_angle, num_sensors, obstacles_gen_f)
    return xml


if __name__ == '__main__':

    test_batch = True

    sensor_angle = 0.6
    num_sensors = 64
    rangefinder_angles = np.linspace(start=-sensor_angle, stop=sensor_angle, num=num_sensors)

    xml = make_xml(sensor_angle, num_sensors)

    with open('forestnav.xml', 'w') as xml_file:
        xml_file.write(xml)

    OBSTACLE_JIGGLE = 0.3
    BATCH_SIZE = 256

    # Create MuJoCo model and data
    mj_model = mujoco.MjModel.from_xml_string(xml)

    # Generate and populate the hills terrain
    # terrain_data = generate_flat_terrain(nrow=32, ncol=32, seed=0, height=0.)
    terrain_data = generate_column_terrain()
    # mj_model.hfield_data[:] = terrain_data
    # mj_model.hfield_data[int(32/2-2):int(32/+2)] = -0.1
    print(f"Generated hills terrain with {len(terrain_data)} data points")
    print(f"Terrain data range: {terrain_data.min():.3f} to {terrain_data.max():.3f}")

    mj_data = mujoco.MjData(mj_model)

    indexer = JointIndexer(mj_model)

    # Transfer model and data to MJX
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # JIT-compile step function
    jit_step = jax.jit(mjx.step)

    if test_batch:
        rng = jax.random.PRNGKey(0)
        rng = jax.random.split(rng, BATCH_SIZE)
        batch = jax.vmap(lambda rng: mjx_data.replace(qpos=uniform_like(rng, mj_data.qpos)))(rng)

        vmap_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        batch = vmap_step(mjx_model, batch)
        print("batch_test_successful")

    # Simulation parameters
    duration = 20.0  # seconds
    framerate = 30  # fps
    n_frames = int(duration * framerate)
    dt = mj_model.opt.timestep
    steps_per_frame = max(1, int(1.0 / (framerate * dt)))

    # Create visualization options
    scene_option = mujoco.MjvOption()
    # Enable heightfield visualization
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True

    # Renderer dimensions - match with the offscreen buffer size
    width, height = 480, 480

    # Prepare data recording
    time_points = []
    rangefinder_data = []
    joint_angle_data = []
    goal_pos_in_body_frame = []
    collision_data = []

    # Reset simulation
    mujoco.mj_resetData(mj_model, mj_data)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # Render and simulate
    frames = []
    with mujoco.Renderer(mj_model, height, width) as renderer:
        # Position the camera to show the terrain better
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth = 45
        cam.elevation = -45
        cam.distance = 8
        cam.lookat = np.array([2., 0, 0.0])

        # orthogonal cam
        # cam = mujoco.MjvCamera()
        # mujoco.mjv_defaultCamera(cam)
        # cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # cam.orthographic = 1  # Enable orthographic projection
        # cam.elevation = -90  # Look straight down
        # cam.azimuth = 0  # Azimuth doesn't matter when looking straight down
        # cam.distance = 10  # Height above scene (adjust as needed)
        # cam.lookat = np.array([2., 0., 0.])  # Center of your scene (x, y, ground level)

        target_vel, target_rotation_vel = 0.6, 1.
        rng = jax.random.PRNGKey(2)

        # Start vehicle higher up to see terrain interaction
        # vehicle_init = jnp.array([0., 0., 0.])
        #
        # qpos_init = vehicle_init
        mjx_data = mjx_data.replace(qvel=jnp.zeros(mjx_data.qvel.shape))

        for i in trange(n_frames):

            ## proportional guidance scheme
            vehicle_pos = mjx_data.sensordata[indexer.vehicle]
            goal_pos = mjx_data.sensordata[3:6]
            goal_vec_in_vehicle_frame = mjx_data.sensordata[6:9]
            collision_sensor = mjx_data.sensordata[9]
            rangefinder = mjx_data.sensordata[10:]

            goal_vec_normalized, distance = normalize_with_norm(goal_vec_in_vehicle_frame)
            ctrl_rotation_vel = - jnp.arcsin(goal_vec_normalized[0])
            ctrl = jnp.array([target_vel, ctrl_rotation_vel])
            mjx_data = mjx_data.replace(ctrl=ctrl)

            # Run multiple steps between frames
            for _ in range(steps_per_frame):
                mjx_data = jit_step(mjx_model, mjx_data)

            # Get data back to CPU
            mj_data = mjx.get_data(mj_model, mjx_data)

            # Record data
            time_points.append(mj_data.time)
            rangefinder_data.append(np.array(rangefinder))
            joint_angle_data.append(mj_data.qpos[2])
            goal_pos_in_body_frame.append(goal_vec_in_vehicle_frame)
            collision_data.append(collision_sensor)

            # Render the frame with custom camera
            # cam.lookat = mjx_data.qpos[indexer.vehicle]
            renderer.update_scene(mj_data, camera=cam, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    # Create video file
    output_filename = "forestnav_v1_hills.mp4"
    media.write_video(output_filename, frames, fps=framerate)
    print(f"Video saved to {output_filename}")

    # Plot rangefinder readings and joint position
    plt.figure(figsize=(12, 10))

    # Plot rangefinder readings
    plt.subplot(4, 1, 1)
    rangefinder_data = np.stack(rangefinder_data)
    plt.imshow(rangefinder_data.T)
    min_idx = 0  # First element (most negative angle)
    center_idx = len(rangefinder_angles) // 2  # Middle element (approximately zero)
    max_idx = len(rangefinder_angles) - 1  # Last element (most positive angle)

    plt.yticks(
        [min_idx, center_idx, max_idx],
        [f'{rangefinder_angles[min_idx]:.2f}', f'{rangefinder_angles[center_idx]:.2f}',
         f'{rangefinder_angles[max_idx]:.2f}']
    )

    plt.xlabel('Time (s)')
    plt.ylabel('Angle (m)')
    plt.title('Rangefinder Readings')
    plt.grid(True)

    # Plot joint angle
    plt.subplot(4, 1, 2)
    plt.plot(time_points, joint_angle_data, label='Joint Angle', color='green', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Joint Angle')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(time_points, goal_pos_in_body_frame, label='Goal Pos', color='green', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('Goal pos in body frame')
    plt.grid(True)

    # Plot collision data
    plt.subplot(4, 1, 4)
    plt.plot(time_points, collision_data, label='Collision', color='red', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Collision (0/1)')
    plt.title('Vehicle Collisions')
    plt.ylim(-0.1, 1.1)  # Since it's binary data
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('rangefinder_data_hills.png')
    plt.show()