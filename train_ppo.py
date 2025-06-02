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

sensor_angle = 0.6
num_sensors = 64

COLLISION_SENSOR = 9
RANGEFINDER_0 = 10

# model, data = forestnav_xml.make_forest_v1(sensor_angle, num_sensors)
#
# rng = jax.random.PRNGKey(0)
# rng = jax.random.split(rng, 4096)
# init_data = jax.vmap(lambda rng: data)(rng)
# step = jax.vmap(mjx.step, in_axes=(None, 0))


def distance(vehicle_pos, goal_pos):
    return jnp.sqrt(jnp.sum((vehicle_pos - goal_pos) ** 2))


class ForestNav(PipelineEnv):
    def __init__(self, sensor_angle, num_sensors, peturb_scale, **kwargs):
        obstacles_gen_f = partial(forestnav_xml.obstacles_grid_xml, [(-1., -1.), (0.5, 0.5)], 0.07)
        xml = forestnav_xml.forestnav_xml(sensor_angle, num_sensors, obstacles_gen_f)

        self.peturb_scale = peturb_scale
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
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
        vehicle_init = jnp.array([-0.95, -0.95, -jnp.pi / 2])
        obstacles_init = (jax.random.uniform(rng, (49-3,)) - 0.5) * self.peturb_scale
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
    parser.add_argument('--entropy_cost', type=float, default=0.0001,
                        help='Entropy regularization coefficient')


    # Parallelization settings
    parser.add_argument('--num_envs', type=int, default=512,
                        help='Number of parallel environments to run')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')

    # Misc
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    return parser


def rollout(env, policy, mp4_filename, seed=0):
    wrapped_env = brax.envs.wrappers.training.wrap(env)

    jit_reset = jax.jit(wrapped_env.reset)
    jit_step = jax.jit(wrapped_env.step)
    vmap_policy = jax.vmap(policy)

    # initialize the state
    batch_size = 2
    rng_key, rng_init = jax.random.split(jax.random.PRNGKey(seed), 2)
    state = jit_reset(jax.random.split(rng_init, batch_size))
    rollout = [jax.tree.map(lambda a: a[0], state.pipeline_state)]
    reward = 0.
    ctrl_seq = []
    xy_pos = []
    reward_seq = []
    collision_seq = []

    # grab a trajectory
    for i in range(args.episode_length):
        rng_key, rng_ctrl = jax.random.split(rng_key)
        rng_ctrl = jax.random.split(rng_key, batch_size)
        ctrl, info = vmap_policy(state.obs, rng_ctrl)
        state = jit_step(state, ctrl)
        reward += state.reward
        reward_seq += [state.reward[0]]
        rollout.append(jax.tree.map(lambda a: a[0], state.pipeline_state))
        ctrl_seq += [ctrl[0]]
        xy_pos += [state.pipeline_state.qpos[0, 0:2]]
        collision_seq += [state.pipeline_state.sensordata[0, COLLISION_SENSOR]]

        if state.done[0]:
            break

    def wandb_log_plot(plt_name, data):
        ctrl_seq = jnp.stack(data)
        fig, ax = plt.subplots()
        ax.plot(ctrl_seq)
        wandb.log({plt_name: fig})

    wandb_log_plot('ctrl_traj', ctrl_seq)
    wandb_log_plot('xy_pos', xy_pos)
    wandb_log_plot('reward', reward_seq)
    wandb_log_plot('collision_sensor', collision_seq)

    media.write_video(mp4_filename, env.render(rollout), fps=1.0 / env.dt)
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
        wandb.log(metrics)

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    # save model
    model_path = '/tmp/mjx_brax_policy'
    model.save_params(model_path, params)

    print("rollout policy")
    params = model.load_params(model_path)

    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    eval_env = envs.get_environment(env_name, sensor_angle=sensor_angle, num_sensors=num_sensors, peturb_scale=args.peturb_scale)

    output_filename = "eval_ppo.mp4"
    reward = rollout(eval_env, jit_inference_fn, output_filename)
    print(f'eval reward: {reward}')

    wandb.log({"eval_video": wandb.Video(output_filename, "eval_video")})
    print('complete')