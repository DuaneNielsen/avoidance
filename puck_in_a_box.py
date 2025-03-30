
from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model
import webbrowser

box_size = 2.0
puck_radius = 0.2

# Define obstacles - each is [x, y, radius, color]
obstacles = [
    [0.2, 0.3, 0.3, "0.2 0.7 0.2 1"],  # Green obstacle
    [-0.5, 0.7, 0.25, "0.7 0.2 0.2 1"],  # Red obstacle
    [0.7, -0.5, 0.2, "0.2 0.2 0.7 1"]  # Blue obstacle
]

# Puck starting position
puck_pos = [-0.2, 0.1, 0]

mjcf_string = """
<mujoco model="2d_puck">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 0" />
  <custom>
    <numeric data="1.0" name="elasticity"/>
    <numeric data="0.0" name="ang_damping"/>
    <numeric data="0.0" name="vel_damping"/>
  </custom>
  <asset>
    <!-- Define materials -->
    <material name="red" rgba="1 0 0 1"/>
    <material name="green" rgba="0 1 0 1"/>
    <material name="blue" rgba="0 0 1 1"/>
    <material name="yellow" rgba="1 1 0 1"/>
    <material name="gray" rgba="0.5 0.5 0.5 0.5"/>
  </asset>

  <worldbody>
    <!-- The puck -->
    <body name="puck" pos="{0} {1} {2}">
      <joint type="free" name="puck_joint"/>
      <geom name="puck_geom" type="sphere" size="{3}" mass="1" rgba="0.7 0.2 0.8 1"/>
      <site name="site_rangefinder0" pos="{3} 0. 0."/>
    </body>
""".format(puck_pos[0], puck_pos[1], puck_pos[2], puck_radius)

# Add obstacles
for i, (x, y, radius, color) in enumerate(obstacles):
    mjcf_string += """
    <!-- Obstacle {0} -->
    <geom name="obstacle_{0}" type="cylinder" pos="{1} {2} 0" 
          size="{3} 0.0001" rgba="{4}" contype="1" conaffinity="1" material="red"/>
""".format(i, x, y, radius, color)

# <geom name="floor" type="box" pos="0 0 -{3}" size="{0} {0} {3}" rgba="0.8 0.8 0.8 1" conaffinity="1"/>
# Add walls using boxes instead of planes
mjcf_string += """
    <!-- Walls of the box -->
    
    <geom name="left_wall" type="box" pos="-{1} 0 0" size="{3} {1} {3}" rgba="0.8 0.8 0.8 1" conaffinity="1" material="gray"/>
    <geom name="right_wall" type="box" pos="{1} 0 0" size="{3} {1} {3}" rgba="0.8 0.8 0.8 1" conaffinity="1" material="gray"/>
    <geom name="front_wall" type="box" pos="0 -{1} 0" size="{1} {3} {3}" rgba="0.8 0.8 0.8 1" conaffinity="1" material="gray"/>
    <geom name="back_wall" type="box" pos="0 {1} 0" size="{1} {3} {3}" rgba="0.8 0.8 0.8 1" conaffinity="1" material="gray"/>
    """.format(box_size, box_size / 2, box_size / 2, 0.01)

mjcf_string += """
  </worldbody>
"""

mjcf_string += """
  <sensor>
    <rangefinder name="rangefinder0" site="site_rangefinder0"/>
  </sensor>
"""

mjcf_string += """
</mujoco>
"""



# Make model, data, and renderer
mj_model = mujoco.MjModel.from_xml_string(mjcf_string)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(mj_data.qpos, type(mj_data.qpos))
print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())

# enable joint and rangefinder visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

frames = []
mujoco.mj_resetData(mj_model, mj_data)
while mj_data.time < duration:
  mujoco.mj_step(mj_model, mj_data)
  if len(frames) < mj_data.time * framerate:
    renderer.update_scene(mj_data, scene_option=scene_option)
    pixels = renderer.render()
    frames.append(pixels)

# Simulate and display video.
# media.show_video(frames, fps=framerate)
media.write_video('./puck_in_a_box_start_0.mp4', frames)

jit_step = jax.jit(mjx.step)

frames = []
mujoco.mj_resetData(mj_model, mj_data)
mjx_data = mjx.put_data(mj_model, mj_data)
while mjx_data.time < duration:
  mjx_data = jit_step(mjx_model, mjx_data)
  if len(frames) < mjx_data.time * framerate:
    mj_data = mjx.get_data(mj_model, mjx_data)
    renderer.update_scene(mj_data, scene_option=scene_option)
    pixels = renderer.render()
    frames.append(pixels)

# media.show_video(frames, fps=framerate)
media.write_video('./puck_in_a_box_1.mp4', frames)

rng = jax.random.PRNGKey(0)
rng = jax.random.split(rng, 4096)
batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (7,))))(rng)

jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
batch = jit_step(mjx_model, batch)

print(batch.qpos)

batched_mj_data = mjx.get_data(mj_model, batch)
print([d.qpos for d in batched_mj_data])

