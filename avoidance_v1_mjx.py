from brax import envs
from brax.envs.base import PipelineEnv, State
import mujoco
from brax.io import mjcf
import jax.numpy as jnp
import mujoco.mjx._src.math as mjxmath

import avoidance_v1 as avoidance
from avoidance_v1 import get_sensor_data_range
from brax.envs.wrappers.training import Wrapper


def read_goal_sensor(model_cpu, data):
    goal_xyz_in_vehicle_frame = get_sensor_data_range(model_cpu, data, avoidance.GOAL_SENSOR_NAME)
    goal_vec_normalized, distance_to_goal = mjxmath.normalize_with_norm(goal_xyz_in_vehicle_frame)
    goal_vec_normalized_x = jnp.clip(goal_vec_normalized[0], -1.0, 1.0)
    angle_to_goal = - jnp.arcsin(goal_vec_normalized_x)
    return distance_to_goal, angle_to_goal


def read_collision_sensors(model_cpu, data):
    left = get_sensor_data_range(model_cpu, data, avoidance.VEHICLE_COLLISION_LEFT_SENSOR_NAME)
    right = get_sensor_data_range(model_cpu, data, avoidance.VEHICLE_COLLISION_RIGHT_SENSOR_NAME)
    front = get_sensor_data_range(model_cpu, data, avoidance.VEHICLE_COLLISION_FRONT_SENSOR_NAME)
    back = get_sensor_data_range(model_cpu, data, avoidance.VEHICLE_COLLISION_BACK_SENSOR_NAME)
    return jnp.abs(left), jnp.abs(right), jnp.abs(front), jnp.abs(back)


def collision_detected(model, data):
    left, right, front, rear = read_collision_sensors(model, data)
    left_collision = left < (avoidance.VEHICLE_LENGTH + avoidance.VEHICLE_COLLISION) * 2
    right_collision = right < (avoidance.VEHICLE_LENGTH + avoidance.VEHICLE_COLLISION) * 2
    front_collision = front < (avoidance.VEHICLE_WIDTH + avoidance.VEHICLE_COLLISION) * 2
    rear_collision = rear < (avoidance.VEHICLE_WIDTH + avoidance.VEHICLE_COLLISION) * 2
    return jnp.any(left_collision | right_collision | front_collision | rear_collision)


class AvoidanceMJX(PipelineEnv):
    def __init__(self, **kwargs):
        self.mj_model_cpu = mujoco.MjModel.from_xml_string(avoidance.xml)
        sys = mjcf.load_model(self.mj_model_cpu)
        super().__init__(sys, **kwargs)

    def list_sensor_names(self):
        """Get a list of all sensor names"""
        sensor_names = []
        for i in range(self.mj_model_cpu.nsensor):
            name = mujoco.mj_id2name(self.mj_model_cpu, mujoco.mjtObj.mjOBJ_SENSOR, i)
            sensor_names.append(name)
        return sensor_names

    def _obs(self, data):
        distance_to_goal, angle_to_goal = read_goal_sensor(self.mj_model_cpu, data)
        RANGEFINDER_0 = mujoco.mj_name2id(self.mj_model_cpu, mujoco.mjtObj.mjOBJ_SENSOR, avoidance.RANGEFINDER_SENSOR_PREFIX + '0')
        rangefinder = data.sensordata[RANGEFINDER_0:]
        rangefinder_norm = jnp.where(rangefinder == -1., avoidance.RANGEFINDER_CUTOFF, rangefinder) / avoidance.RANGEFINDER_CUTOFF
        obs = jnp.concat([angle_to_goal.reshape(1), distance_to_goal.reshape(1)/avoidance.RANGEFINDER_CUTOFF, rangefinder_norm])
        return obs, distance_to_goal

    def reset(self, rng: jnp.ndarray) -> State:
        vehicle_init = jnp.array([0., 0., -jnp.pi / 2])
        qvel = jnp.zeros_like(vehicle_init)
        data = self.pipeline_init(vehicle_init, qvel)

        obs, distance = self._obs(data)
        reward, done, zero = jnp.zeros(3)
        metrics = {'reward': zero}
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:

        # step the simulation
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        # read sensors and get observation
        obs, distance = self._obs(data)

        # calculate the reward
        reward = 10. - distance

        # if we collide with terrain we are done
        collision = collision_detected(self.mj_model_cpu, data)
        done = jnp.where(collision > 0., 1., 0.)

        state.metrics.update(
            reward=reward,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )


envs.register_environment('avoidance_v1', AvoidanceMJX)


class ForwardTurnBrake(Wrapper):
    """Wrapper to convert 3D actions [drive, steer, brake] to 5D actions"""

    def __init__(self, env):
        super().__init__(env)

    def step(self, state, action):
        # Expand 3D action to 5D
        if action.shape[-1] == 3:
            expanded_action = jnp.array([
                action[0],  # drive
                action[1],  # steer
                action[2],  # brake_x
                action[2],  # brake_y
                action[2]  # brake_rot
            ])
        else:
            expanded_action = action

        return self.env.step(state, expanded_action)  # Use self.env, not self._env

    def reset(self, rng):
        return self.env.reset(rng)  # Use self.env, not self._env

    @property
    def action_size(self):
        return 3  # Override to return 3 instead of 5

    @property
    def observation_size(self):
        return self.env.observation_size

    # Forward other important properties
    @property
    def dt(self):
        return self.env.dt

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

if __name__ == '__main__':

    import brax.envs.wrappers.training
    import jax

    seed = 0
    batch_size = 2

    env = envs.get_environment('avoidance_v1')
    wrapped_env = brax.envs.wrappers.training.wrap(env)
    # wrapped_env = env
    # jit_reset = jax.vmap(jax.jit(wrapped_env.reset))
    # jit_step = jax.vmap(jax.jit(wrapped_env.step))

    jit_reset = jax.jit(wrapped_env.reset)
    jit_step = jax.jit(wrapped_env.step)

    rng_key, rng_init = jax.random.split(jax.random.PRNGKey(seed), 2)
    state = jit_reset(jax.random.split(rng_init, batch_size))
    action = jnp.zeros((batch_size, 5))
    state = jit_step(state, action)
    state = jit_step(state, action)


    import jax
    import mediapy
    # import brax.envs.wrappers.training
    import numpy as np


    def default_manual_policy(obs, rng, step_counter):
        """
        Default manual policy for testing

        Args:
            obs: Observation (not used in this simple policy)
            rng: Random key (not used)
            step_counter: Current step number for phased control

        Returns:
            action: [drive, steer, brake]
        """
        episode_length = 300  # Should match the episode_length in rollout

        if step_counter < episode_length // 4:
            # Phase 1: Go forward
            action = jnp.array([1.0, 0.0, 0.0])
        elif step_counter < episode_length // 2:
            # Phase 2: Turn left while going forward
            action = jnp.array([0.5, 1.0, 0.0])
        elif step_counter < 3 * episode_length // 4:
            # Phase 3: Turn right while going forward
            action = jnp.array([0.5, -1.0, 0.0])
        else:
            # Phase 4: Go forward again
            action = jnp.array([1.0, 0.0, 0.0])

        return action, {}  # Return action and empty info dict


    def simple_goal_seeking_policy(obs, rng, step_counter):
        """
        Simple policy that tries to drive toward the goal

        Args:
            obs: Observation array [angle_to_goal, distance_to_goal, rangefinder...]
            rng: Random key
            step_counter: Current step

        Returns:
            action: [drive, steer, brake]
        """
        angle_to_goal = obs[0]
        distance_to_goal = obs[1]

        # Simple proportional control
        drive = jnp.clip(distance_to_goal * 2.0, 0.0, 2.0)  # Drive harder when farther
        steer = jnp.clip(-angle_to_goal * 3.0, -3.0, 3.0)  # Turn toward goal
        brake = 0.0

        # Add some obstacle avoidance using rangefinder
        rangefinder = obs[2:]  # Remaining obs are rangefinder readings
        min_distance = jnp.min(rangefinder)

        # If obstacle is very close, brake and turn
        if min_distance < 0.3:  # 30% of max range
            brake = 0.5
            drive *= 0.5
            # Turn away from closest obstacle
            closest_idx = jnp.argmin(rangefinder)
            # Simple heuristic: turn opposite to where closest obstacle is
            if closest_idx < len(rangefinder) // 2:
                steer = -1.0  # Turn right if obstacle on left
            else:
                steer = 1.0  # Turn left if obstacle on right

        action = jnp.array([drive, steer, brake])
        return action, {}


    def rollout_avoidance(env, policy_fn=None, mp4_filename="avoidance_rollout.mp4",
                          seed=0, batch_size=4, episode_length=200):
        """
        Rollout avoidance environment and render to MP4
        """

        # Use default policy if none provided
        if policy_fn is None:
            policy_fn = default_manual_policy
            print("Using default manual policy (forward -> left -> right -> forward)")

        # Wrap for training (handles batching)
        wrapped_env = brax.envs.wrappers.training.wrap(env)

        jit_reset = jax.jit(wrapped_env.reset)
        jit_step = jax.jit(wrapped_env.step)

        # Initialize state
        rng_key, rng_init = jax.random.split(jax.random.PRNGKey(seed), 2)
        state = jit_reset(jax.random.split(rng_init, batch_size))

        print(f"Initial state shapes:")
        print(f"  obs: {state.obs.shape}")
        print(f"  done: {state.done.shape}")
        print(f"  reward: {state.reward.shape}")

        # Store all batch trajectories
        rollout_batch = [state.pipeline_state]
        total_reward = jnp.zeros(batch_size)

        # Data logging
        ctrl_seq = []
        xy_pos = []
        reward_seq = []
        done_seq = []
        distance_to_goal_seq = []

        print(f"Running {batch_size} parallel rollouts for {episode_length} steps...")

        # Collect trajectories
        for i in range(episode_length):
            # Generate actions using policy
            batch_actions = []
            for batch_idx in range(batch_size):
                rng_key, rng_ctrl = jax.random.split(rng_key)
                action, _ = policy_fn(state.obs[batch_idx], rng_ctrl, i)
                batch_actions.append(action)
            ctrl = jnp.stack(batch_actions)

            # Debug: Check action shape
            if i == 0:
                print(f"Action shape: {ctrl.shape}")

            # Step simulation
            try:
                state = jit_step(state, ctrl)
            except Exception as e:
                print(f"Error at step {i}")
                print(f"State done shape before step: {state.done.shape}")
                print(f"Action shape: {ctrl.shape}")
                raise e

            total_reward += state.reward

            # Store trajectory
            rollout_batch.append(state.pipeline_state)

            # Log data
            reward_seq.append(state.reward)
            ctrl_seq.append(ctrl)
            xy_pos.append(state.pipeline_state.qpos[:, 0:2])
            done_seq.append(state.done)

            # Extract distance to goal from observations
            distance_to_goal = state.obs[:, 1] * avoidance.RANGEFINDER_CUTOFF
            distance_to_goal_seq.append(distance_to_goal)

            # Print progress
            if i % 50 == 0:
                avg_reward = jnp.mean(total_reward)
                any_done = jnp.any(state.done)
                avg_distance = jnp.mean(distance_to_goal)
                print(
                    f"Step {i}: avg_reward={avg_reward:.2f}, any_done={any_done}, avg_dist_to_goal={avg_distance:.2f}")

            # Break if all episodes are done
            if jnp.all(state.done):
                print(f"All episodes finished at step {i}")
                break

        # Rest of the function remains the same...
        # Convert to arrays
        ctrl_seq = jnp.stack(ctrl_seq)
        xy_pos = jnp.stack(xy_pos)
        reward_seq = jnp.stack(reward_seq)
        done_seq = jnp.stack(done_seq)
        distance_to_goal_seq = jnp.stack(distance_to_goal_seq)

        print(f"Rollout complete. Final rewards: {total_reward}")

        # Render all trajectories
        print("Rendering trajectories...")
        all_frames = []

        for batch_idx in range(batch_size):
            print(f"Rendering trajectory {batch_idx + 1}/{batch_size}")

            # Extract individual trajectory
            individual_rollout = []
            for rollout_state in rollout_batch:
                individual_state = jax.tree.map(lambda x: x[batch_idx], rollout_state)
                individual_rollout.append(individual_state)

            # Render this trajectory
            trajectory_frames = env.render(individual_rollout, width=480, height=320)
            all_frames.extend(trajectory_frames)

            # Add separator frames between trajectories
            if batch_idx < batch_size - 1:
                separator_frame = np.zeros_like(trajectory_frames[0])
                all_frames.extend([separator_frame] * 30)

        # Save video
        print(f"Saving video to {mp4_filename}")
        mediapy.write_video(mp4_filename, all_frames, fps=30)

        return {
            'ctrl_seq': ctrl_seq,
            'xy_pos': xy_pos,
            'reward_seq': reward_seq,
            'done_seq': done_seq,
            'distance_to_goal_seq': distance_to_goal_seq,
            'total_reward': total_reward,
            'frames': all_frames
        }

    base_env = envs.get_environment('avoidance_v1')
    env = ForwardTurnBrake(base_env)

    # Example 1: Use default manual policy
    print("=== Running with default manual policy ===")
    results1 = rollout_avoidance(
        env=env,
        policy_fn=None,  # Uses default_manual_policy
        mp4_filename="avoidance_manual.mp4",
        seed=42,
        batch_size=2,
        episode_length=300
    )

    # Example 2: Use goal-s