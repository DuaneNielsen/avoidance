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

    def reset(self, rng: jnp.ndarray) -> State:
        qpos = jnp.array([-1, -1, -jnp.pi / 2])
        qvel = jnp.zeros(3)
        data = self.pipeline_init(qpos, qvel)
        obs = data.sensordata[6:]
        reward, zero = jnp.zeros(2)
        done = jnp.array(0.)
        metrics = {'reward': zero}
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        vehicle_pos = data.sensordata[:3]
        goal_pos = data.sensordata[3:6]
        obs = data.sensordata[6:]
        reward = self.MAX_DISTANCE - distance(vehicle_pos, goal_pos)
        done = jnp.array(data.time > 20., dtype=jnp.float32)

        state.metrics.update(
            reward=reward,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )


envs.register_environment('forestnav_v1', ForestNav)

# env = ForestNav(sensor_angle, num_sensors)
# state = env.reset(jax.random.PRNGKey(0))
# state = env.step(state, action=jnp.array([0.5, 0.8]))

if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import mediapy as media

    # instantiate the environment
    env_name = 'forestnav_v1'
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

    train_fn = partial(
        ppo.train, num_timesteps=20_000_000, num_evals=5, reward_scaling=1.,
        episode_length=1000, normalize_observations=True, action_repeat=5,
        unroll_length=10, num_minibatches=1536, num_updates_per_batch=1,
        discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=3072,
        batch_size=8, seed=0)

    from datetime import datetime

    times = [datetime.now()]
    x_data = []
    y_data = []
    ydataerr = []

    def progress(num_steps, metrics):
        print(f"training {num_steps}: {metrics['eval/episode_reward']}")
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

    eval_env = envs.get_environment(env_name, sensor_angle=0.6, num_sensors=64)

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    # initialize the state
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    # grab a trajectory
    n_steps = 500
    render_every = 2

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)

        if state.done:
            break

    output_filename = "eval_ppo.mp4"
    media.write_video(output_filename, env.render(rollout[::render_every]), fps=1.0 / env.dt / render_every)

    print('complete')