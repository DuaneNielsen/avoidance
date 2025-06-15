import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import forestnav_xml
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model
from brax.base import System
from mujoco import mjx
from functools import partial
import mujoco
from brax import base
from brax import envs
from brax import math
import mujoco.mjx._src.math as mjxmath
import argparse
import brax.envs.wrappers.training
import mediapy as media
from forestnav_xml import uniform_like

sensor_angle = 0.6
num_sensors = 64

COLLISION_SENSOR = 9
RANGEFINDER_0 = 10


def distance(vehicle_pos, goal_pos):
    return jnp.sqrt(jnp.sum((vehicle_pos - goal_pos) ** 2))


class ForestNav(PipelineEnv):
    def __init__(self, sensor_angle, num_sensors, peturb_scale, **kwargs):
        xml = forestnav_xml.make_xml(sensor_angle, num_sensors)

        self.peturb_scale = peturb_scale
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.indexer = forestnav_xml.JointIndexer(self.mj_model)
        sys = mjcf.load_model(self.mj_model)
        self.MAX_DISTANCE = 2.83
        super().__init__(sys, **kwargs)

    def _obs(self, data):
        vehicle_pos = data.sensordata[:3]
        goal_pos = data.sensordata[3:6]
        vehicle_frame_goal_pos = data.sensordata[6:9]
        collision_sensor = data.sensordata[COLLISION_SENSOR]
        rangefinder = data.sensordata[RANGEFINDER_0:]
        rangefinder_norm = jnp.where(rangefinder == -1., self.MAX_DISTANCE, rangefinder) / self.MAX_DISTANCE
        # goal_pos_in_vehicle_frame = data.sensordata[3:6]
        goal_vec_normalized, dist = mjxmath.normalize_with_norm(vehicle_frame_goal_pos)
        goal_vec_normalized_x = jnp.clip(goal_vec_normalized[0], -1.0, 1.0)
        angle_to_goal = - jnp.arcsin(goal_vec_normalized_x)
        # obs = jnp.concat([angle_to_goal.reshape(1), dist.reshape(1) / self.MAX_DISTANCE])
        obs = jnp.concat([angle_to_goal.reshape(1), dist.reshape(1)/self.MAX_DISTANCE, rangefinder_norm])
        # dist = distance(vehicle_pos, goal_pos)
        # obs = jnp.concat([dist.reshape(1)/self.MAX_DISTANCE, rangefinder])
        # obs = rangefinder
        return obs, dist

    def reset(self, rng: jnp.ndarray) -> State:
        vehicle_init = jnp.array([0., 0., -jnp.pi / 2])
        obstacles_init = (uniform_like(rng, self.indexer.obstacles) - 0.5) * self.peturb_scale
        qpos_init = jnp.concatenate([vehicle_init, obstacles_init])
        qvel = jnp.zeros_like(qpos_init)
        data = self.pipeline_init(qpos_init, qvel)
        # jax.debug.print('initial geoms {}', data.geom_xpos)
        # jax.debug.print('final geoms {}', data.geom_xpos)

        obs, distance = self._obs(data)
        reward, done, zero = jnp.zeros(3)
        metrics = {'reward': zero}
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        obs, distance = self._obs(data)
        collision_sensor = data.sensordata[9]
        reward = self.MAX_DISTANCE - distance
        done = jnp.where(collision_sensor > 0., 1., 0.)

        state.metrics.update(
            reward=reward,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )


envs.register_environment('forestnav_v1', ForestNav)


def setup_parser():
    parser = argparse.ArgumentParser(description='Train a PPO agent')

    # Core training parameters
    parser.add_argument('--num_timesteps', type=int, default=40_000_000,
                        help='Total number of environment timesteps to train for')
    parser.add_argument('--num_evals', type=int, default=20,
                        help='Number of evaluation episodes during training')
    parser.add_argument('--reward_scaling', type=float, default=1.0,
                        help='Scaling factor for rewards')
    parser.add_argument('--episode_length', type=int, default=400,
                        help='Maximum length of an episode')

    # Environment settings
    parser.add_argument('--normalize_observations', action='store_false', default=False,
                        help='Whether to normalize observations')
    parser.add_argument('--action_repeat', type=int, default=1,
                        help='Number of times to repeat actions')
    parser.add_argument('--peturb_scale', type=int, default=0.3,
                        help='Max distance to randomly peturb the obstacles')

    # PPO algorithm parameters

    parser.add_argument('--num_minibatches', type=int, default=512,
                        help='Number of minibatches to use per update')
    parser.add_argument('--unroll_length', type=int, default=40,
                        help='Length of segments to train on')
    parser.add_argument('--num_updates_per_batch', type=int, default=4,
                        help='Number of passes over each batch')
    parser.add_argument('--discounting', type=float, default=0.99,
                        help='Discount factor for future rewards')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                        help='Learning rate for optimizer')
    parser.add_argument('--entropy_cost', type=float, default=0.01,
                        help='Entropy regularization coefficient')


    # Parallelization settings
    parser.add_argument('--num_envs', type=int, default=128,
                        help='Number of parallel environments to run')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')

    # Misc
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    return parser

import numpy as np


def rollout(env, policy, mp4_filename, seed=0, batch_size=4):
    wrapped_env = brax.envs.wrappers.training.wrap(env)

    jit_reset = jax.jit(wrapped_env.reset)
    jit_step = jax.jit(wrapped_env.step)
    vmap_policy = jax.vmap(policy)

    # initialize the state
    rng_key, rng_init = jax.random.split(jax.random.PRNGKey(seed), 2)
    state = jit_reset(jax.random.split(rng_init, batch_size))

    # Store all batch trajectories
    rollout_batch = [state.pipeline_state]
    reward = jnp.zeros(batch_size)

    # Collect data for ALL batch members
    ctrl_seq = []  # Will be shape [timesteps, batch_size, action_dim]
    xy_pos = []  # Will be shape [timesteps, batch_size, 2]
    reward_seq = []  # Will be shape [timesteps, batch_size]
    collision_seq = []  # Will be shape [timesteps, batch_size]

    # grab trajectories for all batch members
    for i in range(args.episode_length):
        rng_key, rng_ctrl = jax.random.split(rng_key)
        rng_ctrl = jax.random.split(rng_key, batch_size)
        ctrl, info = vmap_policy(state.obs, rng_ctrl)
        state = jit_step(state, ctrl)
        reward += state.reward

        # Store the entire batch state
        rollout_batch.append(state.pipeline_state)

        # Log data from ALL trajectories
        reward_seq.append(state.reward)  # [batch_size]
        ctrl_seq.append(ctrl)  # [batch_size, action_dim]
        xy_pos.append(state.pipeline_state.qpos[:, 0:2])  # [batch_size, 2]
        collision_seq.append(state.pipeline_state.sensordata[:, COLLISION_SENSOR])  # [batch_size]

        if jnp.any(state.done):
            break

    # Convert lists to arrays for easier plotting
    ctrl_seq = jnp.stack(ctrl_seq)  # [timesteps, batch_size, action_dim]
    xy_pos = jnp.stack(xy_pos)  # [timesteps, batch_size, 2]
    reward_seq = jnp.stack(reward_seq)  # [timesteps, batch_size]
    collision_seq = jnp.stack(collision_seq)  # [timesteps, batch_size]

    # Render all trajectories sequentially
    all_frames = []
    for batch_idx in range(batch_size):
        # Extract trajectory for this batch member
        individual_rollout = []
        for rollout_state in rollout_batch:
            individual_state = jax.tree.map(lambda x: x[batch_idx], rollout_state)
            individual_rollout.append(individual_state)

        # Render this trajectory
        trajectory_frames = env.render(individual_rollout)
        all_frames.extend(trajectory_frames)

        # Add a few black frames as separator between trajectories
        if batch_idx < batch_size - 1:
            black_frame = np.zeros_like(trajectory_frames[0])
            all_frames.extend([black_frame] * 10)  # 10 frames of black

    def wandb_log_plot_batch(plt_name, data, ylabel="Value", labels=None):
        """Plot data for all batch members on the same plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        if labels is None:
            labels = [f"Trajectory {i}" for i in range(batch_size)]

        # If data has multiple dimensions (like ctrl with action_dim > 1), plot each dimension
        if len(data.shape) == 3:  # [timesteps, batch_size, feature_dim]
            for feature_idx in range(data.shape[2]):
                for batch_idx in range(batch_size):
                    ax.plot(data[:, batch_idx, feature_idx],
                            label=f"{labels[batch_idx]} (dim {feature_idx})",
                            alpha=0.7)
        else:  # [timesteps, batch_size]
            for batch_idx in range(batch_size):
                ax.plot(data[:, batch_idx], label=labels[batch_idx], alpha=0.7)

        ax.set_xlabel('Timestep')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{plt_name} - All Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)

        wandb.log({plt_name: fig})
        plt.close(fig)  # Important to close to free memory

    def wandb_log_xy_trajectory():
        """Special plot for XY positions showing trajectory paths"""
        fig, ax = plt.subplots(figsize=(10, 8))

        for batch_idx in range(batch_size):
            x_traj = xy_pos[:, batch_idx, 0]
            y_traj = xy_pos[:, batch_idx, 1]

            # Plot trajectory
            ax.plot(x_traj, y_traj, label=f"Trajectory {batch_idx}", alpha=0.7, linewidth=2)

            # Mark start and end points
            ax.scatter(x_traj[0], y_traj[0], marker='o', s=100,
                       label=f"Start {batch_idx}" if batch_idx == 0 else "", color='green')
            ax.scatter(x_traj[-1], y_traj[-1], marker='x', s=100,
                       label=f"End {batch_idx}" if batch_idx == 0 else "", color='red')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Vehicle Trajectories in XY Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        wandb.log({"xy_trajectories": fig})
        plt.close(fig)

    # Create all the plots
    wandb_log_plot_batch('ctrl_traj', ctrl_seq, ylabel="Control Signal")
    wandb_log_plot_batch('reward', reward_seq, ylabel="Reward")
    wandb_log_plot_batch('collision_sensor', collision_seq, ylabel="Collision Sensor")
    wandb_log_xy_trajectory()

    media.write_video(mp4_filename, all_frames, fps=1.0 / env.dt)
    return reward


if __name__ == '__main__':

    import wandb

    parser = setup_parser()
    parser.add_argument('--dev', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # instantiate the environment
    env_name = 'forestnav_v1'

    run = wandb.init(
        project=f"{env_name}_ppo",
        config=args,
        save_code=True,
        settings=wandb.Settings(code_dir=".")
    )

    env = envs.get_environment(env_name,
                               sensor_angle=sensor_angle,
                               num_sensors=num_sensors,
                               peturb_scale=args.peturb_scale
                               )

    if args.dev:
        # define the jit reset/step functions

        def policy(obs, rng):
            return jnp.array([0.6, obs[0]]), None

        print("dev mode - testing environment")
        wandb.run.tags = wandb.run.tags + ("dev",)
        reward = rollout(env, policy, 'dev_rollout_seed_0.mp4', seed=0)
        reward = rollout(env, policy, 'dev_rollout_seed_1.mp4', seed=1)
        print(f"dev reward: {reward}")

    if args.dev:
        print('dev mode - training')
        train_fn = partial(
            ppo.train, num_timesteps=5000, num_evals=5, reward_scaling=800.,
            episode_length=500, normalize_observations=True, action_repeat=5,
            unroll_length=10, num_minibatches=32, num_updates_per_batch=1,
            discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=32,
            batch_size=8, seed=0, wrap_env=True, wrap_env_fn=brax.envs.wrappers.training.wrap)
    else:
        print('start training')
        train_fn = partial(
            ppo.train, num_timesteps=args.num_timesteps, num_evals=args.num_evals, reward_scaling=args.reward_scaling,
            episode_length=args.episode_length, normalize_observations=args.normalize_observations, action_repeat=args.action_repeat,
            unroll_length=args.unroll_length, num_minibatches=args.num_minibatches, num_updates_per_batch=args.num_updates_per_batch,
            discounting=args.discounting, learning_rate=args.learning_rate, entropy_cost=args.entropy_cost, num_envs=args.num_envs,
            batch_size=args.batch_size, seed=args.seed, wrap_env=True, wrap_env_fn=brax.envs.wrappers.training.wrap)

    def progress(num_steps, metrics):
        print(f"training {num_steps}: {metrics['eval/episode_reward']}")
        print(metrics)
        wandb.log(metrics)

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    # save model
    model_path = f'{run.dir}/mjx_brax_policy.pt'
    model.save_params(model_path, params)

    print("rollout policy")
    params = model.load_params(model_path)

    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    eval_env = envs.get_environment(env_name, sensor_angle=sensor_angle, num_sensors=num_sensors, peturb_scale=args.peturb_scale)

    output_filename = "eval_ppo.mp4"
    reward = rollout(eval_env, jit_inference_fn, output_filename, batch_size=16)
    print(f'eval reward: {reward}')

    wandb.log({"eval_video": wandb.Video(output_filename, "eval_video")})
    print('complete')