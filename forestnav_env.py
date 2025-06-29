import jax
import jax.numpy as jnp
import mujoco
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.io import html, mjcf, model
from brax.base import System
from mujoco import mjx
from brax import base
from brax import envs
from brax import math
import mujoco.mjx._src.math as mjxmath
import forestnav_xml
from forestnav_xml import uniform_like

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


# Register the environment
envs.register_environment('forestnav_v1', ForestNav)