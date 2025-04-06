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


def obstacles_grid_xml(skip_pos: List[float]=None, radius=0.07):
    if skip_pos is None:
        skip_pos = [(0., 0.)]

    xml = ""

    obstacles = []
    for x in np.linspace(-1., 1., 5):
        for y in np.linspace(-1., 1, 5):
            skip = False
            for skip_x, skip_y in skip_pos:
                if x == skip_x and y == skip_y:
                    skip = True
                    break
            if not skip:
                obstacles.append([x, y, radius])

    for i, (x, y, radius) in enumerate(obstacles):
        xml += f"""
        <!-- Obstacle {i} -->
        <geom name="obstacle_{i}" type="sphere" pos="{x} {y} 0" 
              size="{radius}" contype="1" conaffinity="1" material="obstacle_material"/>
    """
    return xml


def forestnav_xml(
        sensor_angle=0.6,
        num_sensors=128,
        obstacles_xml_f: Callable = None
):

    xml = """
    <mujoco model="forestnav_v1">
      <compiler autolimits="true"/>
      
      <option integrator="implicitfast"/>
    
      <asset>
        <material name="body_material" rgba="0.2 0.8 0.2 1"/>
        <material name="obstacle_material" rgba="0.8 0.2 0.2 1"/>
        <material name="goal_material" rgba="0.3 0.9 0.2 1"/>
        <material name="floor_material" rgba="0.3 0.3 0.3 1"/>
      </asset>
    
      <option timestep="0.02">
        <!--flag contact="disable" /-->
      </option>
    
      <default>
        <joint damping="0.25" stiffness="0.25"/>
      </default>
      
      <worldbody>
    """

    xml += f"""
        <site name="origin"/>
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    
        <!-- stacked joint: hinge + slide -->
        <body pos="0.0 0 0" name="vehicle">
          <joint name="x_joint" type="slide" axis="1. 0. 0." range="-1 1"/>
          <joint name="y_joint" type="slide" axis="0. 1. 0." range="-1 1"/>
          <joint name="rot_joint" type="hinge" axis="0 0 1."/>
          <site name="velocity_site" pos="0 0 0" size="0.01"/>
          <frame pos="0 0.01 0" quat="-1 1 0 0">
          <geom name="vehicle_body" type="box" pos="0 0 0" size=".0168 .01 .005" mass="0.1"/>
          <site name="vehicle_collision_site" type="box" pos="0 0 0" size=".0168 .01 .005" mass="0.0"/>
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
        <geom name="goal" pos="1. 1. 0" type="sphere" size="0.07" material="goal_material"/>
    """

    xml += obstacles_xml_f()

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


def make_forest_v1(sensor_angle, num_sensors):
    """

    :param sensor_angle:
    :param num_sensors:
    :return: mjx_model, mjx_data
    """
    obstacles_gen_f = partial(obstacles_grid_xml, [(-1., -1.), (1., 1.)], 0.07)
    xml = forestnav_xml(sensor_angle, num_sensors, obstacles_gen_f)

    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)

    # Transfer model and data to MJX
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)
    return mjx_model, mjx_data


if __name__ == '__main__':

    test_batch = True

    sensor_angle = 0.6
    num_sensors = 64
    rangefinder_angles = np.linspace(start=-sensor_angle, stop=sensor_angle, num=num_sensors)

    obstacles_gen_f = partial(obstacles_grid_xml, [(-1., -1.), (1., 1.)], 0.07)

    xml = forestnav_xml(sensor_angle, num_sensors, obstacles_gen_f)

    # Create MuJoCo model and data
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)

    # Transfer model and data to MJX
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # JIT-compile step function
    jit_step = jax.jit(mjx.step)

    if test_batch:
        rng = jax.random.PRNGKey(0)
        rng = jax.random.split(rng, 4096)
        batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (3,))))(rng)

        vmap_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        batch = vmap_step(mjx_model, batch)
        print("batch_test_successful")

    # Simulation parameters
    duration = 10.0  # seconds
    framerate = 30  # fps
    n_frames = int(duration * framerate)
    dt = mj_model.opt.timestep
    steps_per_frame = max(1, int(1.0 / (framerate * dt)))

    # Create visualization options
    scene_option = mujoco.MjvOption()
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    scene_option.frame = mujoco.mjtFrame.mjFRAME_SITE
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True

    # Renderer dimensions - match with the offscreen buffer size
    width, height = 480, 480  # swapped to match the XML sizes

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
        # Position the camera for better view
        cam = mujoco.MjvCamera()
        cam.azimuth = 90
        cam.elevation = -50
        cam.distance = 3.5
        cam.lookat = np.array([0, 0, 0.2])

        target_vel, target_rotation_vel = 0.6, 1.

        mjx_data = mjx_data.replace(qpos=jnp.array([-1, -1, -jnp.pi/2]))

        for i in trange(n_frames):

            vehicle_pos = mjx_data.sensordata[:3]
            goal_pos = mjx_data.sensordata[3:6]
            goal_vec_in_vehicle_frame = mjx_data.sensordata[6:9]
            collision_sensor = mjx_data.sensordata[9]
            rangefinder = mjx_data.sensordata[10:]

            goal_vec_normalized, distance = normalize_with_norm(goal_vec_in_vehicle_frame)
            ctrl_rotation_vel = - jnp.arcsin(goal_vec_normalized[0])
            ctrl = jax.numpy.array([target_vel, ctrl_rotation_vel])
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

            # Render the frame
            renderer.update_scene(mj_data, camera=cam, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    # Create video file
    output_filename = "forestnav_v1.mp4"
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
    plt.savefig('rangefinder_data.png')
    plt.show()
