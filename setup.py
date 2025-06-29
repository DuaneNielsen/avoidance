from setuptools import setup, find_packages
import os

def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_file = 'requirements.txt'
    
    if not os.path.exists(requirements_file):
        print(f"Warning: {requirements_file} not found. Run 'python generate_requirements.py' first.")
        return []
    
    requirements = []
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                requirements.append(line)
    
    return requirements

# Read dependencies from requirements.txt
install_requires = read_requirements()

setup(
    name="forestnav",
    version="0.1.0",
    description="Reinforcement learning project for autonomous navigation with MuJoCo physics simulation",
    author="ForestNav Project",
    packages=find_packages(),
    py_modules=['avoidance_v1', 'avoidance_v1_mjx', 'quaterion', 'train_ppo', 'forestnav_env', 'forestnav_xml', 'train_jax_ppo'],
    install_requires=install_requires,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)