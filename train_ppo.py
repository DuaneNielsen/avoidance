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
        rangefinder = data.sensordata[9:]
        # goal_pos_in_vehicle_frame = data.sensordata[3:6]
        goal_vec_normalized, dist = mjxmath.normalize_with_norm(vehicle_frame_goal_pos)
        goal_vec_normalized_x = jnp.clip(goal_vec_normalized[0], -1.0, 1.0)
        angle_to_goal = - jnp.arcsin(goal_vec_normalized_x)
        obs = jnp.concat([angle_to_goal.reshape(1), dist.reshape(1) / self.MAX_DISTANCE])
        # obs = jnp.concat([angle_to_goal.reshape(1), dist.reshape(1)/self.MAX_DISTANCE, rangefinder])
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
        reward = self.MAX_DISTANCE - distance
        done = jnp.array(data.time > 20., dtype=jnp.float32)

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
    parser.add_argument('--num_timesteps', type=int, default=2_000_000,
                        help='Total number of environment timesteps to train for')
    parser.add_argument('--num_evals', type=int, default=20,
                        help='Number of evaluation episodes during training')
    parser.add_argument('--reward_scaling', type=float, default=800.0,
                        help='Scaling factor for rewards')
    parser.add_argument('--episode_length', type=int, default=1000,
                        help='Maximum length of an episode')

    # Environment settings
    parser.add_argument('--normalize_observations', action='store_false', default=False,
                        help='Whether to normalize observations')
    parser.add_argument('--action_repeat', type=int, default=5,
                        help='Number of times to repeat actions')

    # PPO algorithm parameters
    parser.add_argument('--unroll_length', type=int, default=10,
                        help='Length of segments to train on')
    parser.add_argument('--num_minibatches', type=int, default=256,
                        help='Number of minibatches to use per update')
    parser.add_argument('--num_updates_per_batch', type=int, default=1,
                        help='Number of passes over each batch')
    parser.add_argument('--discounting', type=float, default=0.97,
                        help='Discount factor for future rewards')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate for optimizer')
    parser.add_argument('--entropy_cost', type=float, default=1e-3,
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


if __name__ == '__main__':


    from matplotlib import pyplot as plt
    import mediapy as media
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

    env = envs.get_environment(env_name, sensor_angle=0.6, num_sensors=64)

    # define the jit reset/step functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    print("test environment")
    # initialize the state
    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state.pipeline_state]

    # grab a trajectory
    for i in range(600):
        ctrl = -0.1 * jnp.ones(env.sys.nu)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)

    output_filename = "rollout_ppo.mp4"
    media.write_video(output_filename, env.render(rollout), fps=1.0 / env.dt)

    print('start training')

    if args.dev:
        print('dev mode')
        train_fn = partial(
            ppo.train, num_timesteps=5000, num_evals=5, reward_scaling=800.,
            episode_length=500, normalize_observations=True, action_repeat=5,
            unroll_length=10, num_minibatches=32, num_updates_per_batch=1,
            discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=32,
            batch_size=8, seed=0)
    else:
        train_fn = partial(
            ppo.train, num_timesteps=2_000_000, num_evals=20, reward_scaling=800.,
            episode_length=1000, normalize_observations=False, action_repeat=5,
            unroll_length=10, num_minibatches=256, num_updates_per_batch=1,
            discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=128,
            batch_size=8, seed=0)


    from datetime import datetime

    times = [datetime.now()]
    x_data = []
    y_data = []
    ydataerr = []

    def progress(num_steps, metrics):
        print(f"training {num_steps}: {metrics['eval/episode_reward']}")
        wandb.log(metrics)
      # times.append(datetime.now())
      # x_data.append(num_steps)
      # y_data.append(metrics['eval/episode_reward'])
      # ydataerr.append(metrics['eval/episode_reward_std'])
      #
      # plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
      # # plt.ylim([min_y, max_y])
      #
      # plt.xlabel('# environment steps')
      # plt.ylabel('reward per episode')
      # plt.title(f'y={y_data[-1]:.3f}')
      #
      # plt.errorbar(
      #     x_data, y_data, yerr=ydataerr)
      # plt.show()

    make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

    # save model
    model_path = '/tmp/mjx_brax_policy'
    model.save_params(model_path, params)
    params = model.load_params(model_path)

    print("rollout policy")

    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    eval_env = envs.get_environment(env_name, sensor_angle=sensor_angle, num_sensors=num_sensors)

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    # initialize the state
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    # grab a trajectory
    render_every = 2

    for i in range(args.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)

        if state.done:
            break

    output_filename = "eval_ppo.mp4"
    media.write_video(output_filename, env.render(rollout[::render_every]), fps=1.0 / env.dt / render_every)

    wandb.log({"eval_video": wandb.Video(output_filename, "eval_video")})
    print('complete')