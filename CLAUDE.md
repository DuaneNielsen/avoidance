# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ForestNav is a reinforcement learning project for autonomous navigation in complex terrain using MuJoCo physics simulation. The project implements a 2D vehicle that navigates through heightfield terrain using rangefinder sensors and obstacle avoidance algorithms.

## Core Architecture

### Main Components

1. **Environment Classes**
   - `AvoidanceMJX` (avoidance_v1_mjx.py): Brax-based environment for MJX physics simulation
   - `ForestNav` (train_ppo.py): Main environment for PPO training with configurable sensors

2. **Vehicle Simulation**
   - `avoidance_v1.py`: Core vehicle dynamics with heightfield terrain and sensor arrays
   - `forestnav_xml.py`: XML generation for MuJoCo scenes with dynamic obstacle placement

3. **Training Pipeline**
   - `train_ppo.py`: PPO agent training using Brax framework
   - Extensive Weights & Biases logging for experiment tracking

### Key Libraries

- **Brax**: Google's JAX-based physics engine for RL environments
- **MuJoCo/MJX**: Physics simulation with JAX acceleration
- **JAX**: Numerical computing for ML training
- **MediaPy**: Video generation for training visualization

## Development Commands

### Running Simulations

```bash
# Basic vehicle avoidance simulation
python avoidance_v1.py

# MJX-accelerated simulation
python avoidance_v1_mjx.py

# PPO training
python train_ppo.py
```

### Environment Setup

The project uses Python with JAX ecosystem dependencies. Key requirements include:
- JAX and JAX-lib
- MuJoCo and MuJoCo-MJX
- Brax
- MediaPy for video output
- Weights & Biases for experiment tracking

### Sensor Configuration

The vehicle uses configurable rangefinder arrays:
- `SENSOR_ANGLE_DEGREES`: Field of view for sensor array
- `NUM_SENSORS`: Number of rangefinder sensors
- `RANGEFINDER_CUTOFF`: Maximum sensor range

Vehicle collision detection uses four-point sensors (front, back, left, right).

## File Structure

### Core Simulation Files
- `avoidance_v1.py`: Main vehicle simulation with MuJoCo viewer
- `avoidance_v1_mjx.py`: Brax environment wrapper for training
- `forestnav_xml.py`: Dynamic XML scene generation
- `train_ppo.py`: PPO training implementation

### Study Directory
- `study/`: Experimental and development files
  - `heightfield_study/`: Terrain generation experiments
  - `minimal_examples/`: Basic MuJoCo examples
  - Various sensor and visualization experiments

### Generated Content
- `*.xml`: MuJoCo scene definitions
- `*.mp4`: Training and evaluation videos
- `wandb/`: Weights & Biases experiment logs
- `terrain_video/`: Terrain visualization outputs

## Simulation Configuration

### Vehicle Parameters
- Vehicle dimensions and collision geometry defined in `avoidance_v1.py`
- Configurable start positions and goal locations
- Adjustable sensor arrays and collision detection

### Terrain Generation
- Heightfield terrain with configurable complexity
- Dynamic obstacle placement via `forestnav_xml.py`
- Perlin noise heightfield generation available

## Training Notes

- Uses PPO algorithm with Brax training framework
- Heavy use of Weights & Biases for experiment tracking
- Training generates videos for evaluation
- Sensor data includes rangefinder arrays and goal direction vectors
- Collision detection integrated into reward function

## Video Generation

The project generates training videos using MediaPy. Videos show:
- Vehicle navigation through terrain
- Sensor visualization
- Goal-seeking behavior
- Collision avoidance

## Key Constants

From `avoidance_v1.py`:
- `SENSOR_ANGLE_DEGREES = 64`
- `NUM_SENSORS = 64`
- `RANGEFINDER_CUTOFF = 4.83`
- `GOAL_POS = "10. 0. 0.2"`