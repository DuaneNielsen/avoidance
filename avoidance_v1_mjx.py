from brax import envs
from brax.envs.base import PipelineEnv, State
import mujoco
from brax.io import mjcf
import jax.numpy as jnp
import mujoco.mjx._src.math as mjxmath

import avoidance_v1 as avoidance
from avoidance_v1 import get_sensor_data_range


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
    return left_collision | right_collision | front_collision | rear_collision


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
        collision_sensor = collision_detected(self.mj_model_cpu, data)
        done = jnp.where(collision_sensor > 0., 1., 0.)

        state.metrics.update(
            reward=reward,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )


envs.register_environment('avoidance_v1', AvoidanceMJX)

if __name__ == '__main__':

    import brax.envs.wrappers.training
    import jax

    seed = 0
    batch_size = 2

    env = envs.get_environment('avoidance_v1')
    print(env.list_sensor_names())
    wrapped_env = brax.envs.wrappers.training.wrap(env)
    jit_reset = jax.jit(wrapped_env.reset)
    jit_step = jax.jit(wrapped_env.step)

    rng_key, rng_init = jax.random.split(jax.random.PRNGKey(seed), 2)
    state = jit_reset(jax.random.split(rng_init, batch_size))

