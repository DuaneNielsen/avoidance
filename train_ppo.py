import jax
import jax.numpy as jnp
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


# model, data = forestnav_xml.make_forest_v1(sensor_angle, num_sensors)
#
# rng = jax.random.PRNGKey(0)
# rng = jax.random.split(rng, 4096)
# init_data = jax.vmap(lambda rng: data)(rng)
# step = jax.vmap(mjx.step, in_axes=(None, 0))


def distance(vehicle_pos, goal_pos):
    return jnp.sqrt(jnp.sum((vehicle_pos - goal_pos) ** 2))


class ForestNav(PipelineEnv):
    def __init__(self, sensor_angle, num_sensors, **kwargs):
        obstacles_gen_f = partial(forestnav_xml.obstacles_grid_xml, [(-1., -1.), (1., 1.)], 0.07)
        xml = forestnav_xml.forestnav_xml(sensor_angle, num_sensors, obstacles_gen_f)

        mj_model = mujoco.MjModel.from_xml_string(xml)
        sys = mjcf.load_model(mj_model)
        self.MAX_DISTANCE = 2.83
        super().__init__(sys, **kwargs)

    def _obs(self, data):
        vehicle_pos = data.sensordata[:3]
        goal_pos = data.sensordata[3:6]
        vehicle_frame_goal_pos = data.sensordata[6:9]
        collision_sensor = data.sensordata[9]
        rangefinder = data.sensordata[10:]
        # goal_pos_in_vehicle_frame = data.sensordata[3:6]
        goal_vec_normalized, dist = mjxmath.normalize_with_norm(vehicle_frame_goal_pos)
        goal_vec_normalized_x = jnp.clip(goal_vec_normalized[0], -1.0, 1.0)
        angle_to_goal = - jnp.arcsin(goal_vec_normalized_x)
        # obs = jnp.concat([angle_to_goal.reshape(1), dist.reshape(1) / self.MAX_DISTANCE])
        obs = jnp.concat([angle_to_goal.reshape(1), dist.reshape(1)/self.MAX_DISTANCE, rangefinder])
        # dist = distance(vehicle_pos, goal_pos)
        # obs = jnp.concat([dist.reshape(1)/self.MAX_DISTANCE, rangefinder])
        # obs = rangefinder
        return obs, dist

    def reset(self, rng: jnp.ndarray) -> State:
        qpos = jnp.array([-1, -1, -jnp.pi / 2])
        qvel = jnp.zeros(3)
        data = self.pipeline_init(qpos, qvel)
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
    parser.add_argument('--num_timesteps', type=int, default=20_000_000,
                        help='Total number of environment timesteps to train for')
    parser.add_argument('--num_evals', type=int, default=20,
                        help='Number of evaluation episodes during training')
    parser.add_argument('--reward_scaling', type=float, default=1.0,
                        help='Scaling factor for rewards')
    parser.add_argument('--episode_length', type=int, default=600,
                        help='Maximum length of an episode')

    # Environment settings
    parser.add_argument('--normalize_observations', action='store_false', default=False,
                        help='Whether to normalize observations')
    parser.add_argument('--action_repeat', type=int, default=1,
                        help='Number of times to repeat actions')

    # PPO algorithm parameters
    parser.add_argument('--unroll_length', type=int, default=20,
                        help='Length of segments to train on')
    parser.add_argument('--num_minibatches', type=int, default=256,
                        help='Number of minibatches to use per update')
    parser.add_argument('--num_updates_per_batch', type=int, default=1,
                        help='Number of passes over each batch')
    parser.add_argument('--discounting', type=float, default=0.99,
                        help='Discount factor for future rewards')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--entropy_cost', type=float, default=0.001,
                        help='Entropy regularization coefficient')

    # Parallelization settings
    parser.add_argument('--num_envs', type=int, default=256,
                        help='Number of parallel environments to run')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')

    # Misc
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    return parser


def rollout(env, policy, mp4_filename):
    wrapped_env = brax.envs.wrappers.training.wrap(env)

    jit_reset = jax.jit(wrapped_env.reset)
    jit_step = jax.jit(wrapped_env.step)
    vmap_policy = jax.vmap(policy)

    # initialize the state
    batch_size = 2
    rng_key, rng_init = jax.random.split(jax.random.PRNGKey(0), 2)
    state = jit_reset(jax.random.split(rng_init, batch_size))
    rollout = [jax.tree.map(lambda a: a[0], state.pipeline_state)]
    reward = 0.

    # grab a trajectory
    for i in range(args.episode_length):
        rng_key, rng_ctrl = jax.random.split(rng_key)
        rng_ctrl = jax.random.split(rng_key, batch_size)
        ctrl, info = vmap_policy(state.obs, rng_ctrl)
        state = jit_step(state, ctrl)
        reward += state.reward
        rollout.append(jax.tree.map(lambda a: a[0], state.pipeline_state))

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

    env = envs.get_environment(env_name, sensor_angle=sensor_angle, num_sensors=num_sensors)

    if args.dev:
        # define the jit reset/step functions

        def policy(obs, rng):
            return jnp.array([0.6, obs[0]]), None

        print("dev mode - testing environment")
        reward = rollout(env, policy, 'dev_rollout.mp4')
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

    from datetime import datetime

    times = [datetime.now()]
    x_data = []
    y_data = []
    ydataerr = []

    def progress(num_steps, metrics):
        print(f"training {num_steps}: {metrics['eval/episode_reward']}")
        wandb.log(metrics)

    make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

    # save model
    model_path = '/tmp/mjx_brax_policy'
    model.save_params(model_path, params)

    if args.dev:
        print("rollout policy")
        params = model.load_params(model_path)

        inference_fn = make_inference_fn(params)
        jit_inference_fn = jax.jit(inference_fn)

        eval_env = envs.get_environment(env_name, sensor_angle=sensor_angle, num_sensors=num_sensors)

        reward = rollout(env, jit_inference_fn, "eval_ppo.mp4")
        print(f'eval reward: {reward}')

    # jit_reset = jax.jit(eval_env.reset)
    # jit_step = jax.jit(eval_env.step)
    #
    # # initialize the state
    # rng = jax.random.PRNGKey(0)
    # state = jit_reset(rng)
    # rollout = [state.pipeline_state]
    #
    # # grab a trajectory
    # render_every = 2
    #
    # for i in range(args.episode_length):
    #     act_rng, rng = jax.random.split(rng)
    #     ctrl, _ = jit_inference_fn(state.obs, act_rng)
    #     state = jit_step(state, ctrl)
    #     rollout.append(state.pipeline_state)
    #
    #     if state.done:
    #         break
    #
    # output_filename = "eval_ppo.mp4"
    # media.write_video(output_filename, env.render(rollout[::render_every]), fps=1.0 / env.dt / render_every)
    #
    # wandb.log({"eval_video": wandb.Video(output_filename, "eval_video")})
    # print('complete')