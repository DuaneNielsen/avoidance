from setuptools import setup, find_packages
import ast
import os

def get_imports_from_file(filepath):
    """Extract all import statements from a Python file."""
    imports = set()
    
    with open(filepath, 'r') as file:
        try:
            tree = ast.parse(file.read())
        except SyntaxError:
            return imports
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Get the top-level package name
                package = alias.name.split('.')[0]
                imports.add(package)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Get the top-level package name
                package = node.module.split('.')[0]
                imports.add(package)
    
    return imports

def get_all_dependencies():
    """Get dependencies from the main simulation files."""
    target_files = ['avoidance_v1.py', 'avoidance_v1_mjx.py']
    all_imports = set()
    
    for filepath in target_files:
        if os.path.exists(filepath):
            file_imports = get_imports_from_file(filepath)
            all_imports.update(file_imports)
            print(f"Imports from {filepath}: {sorted(file_imports)}")
    
    # Filter out standard library modules and local imports
    stdlib_modules = {
        'time', 'os', 'sys', 'math', 'random', 'collections', 'functools', 
        'itertools', 'operator', 'typing', 'tempfile', 'shutil', 'ast'
    }
    
    # Remove standard library and local modules
    external_imports = all_imports - stdlib_modules - {'avoidance_v1', 'quaterion'}
    
    return sorted(external_imports)

# Get dependencies automatically
detected_deps = get_all_dependencies()
print(f"Detected external dependencies: {detected_deps}")

# Map import names to PyPI package names where they differ
package_mapping = {
    'cv2': 'opencv-python==4.11.0.86',
    'mujoco': 'mujoco>=3.0.0',
    'brax': 'brax',
    'jax': 'jax[cpu]',  # or jax[cuda] for GPU support
    'mediapy': 'mediapy',
    'numpy': 'numpy>=1.21.2',
    'tqdm': 'tqdm',
}

# Convert detected imports to proper package names
install_requires = []
for dep in detected_deps:
    if dep in package_mapping:
        install_requires.append(package_mapping[dep])
    else:
        install_requires.append(dep)

# Add the local quaterion module to the packages
packages = find_packages()

setup(
    name="forestnav",
    version="0.1.0",
    description="Reinforcement learning project for autonomous navigation with MuJoCo physics simulation",
    author="ForestNav Project",
    packages=packages,
    py_modules=['avoidance_v1', 'avoidance_v1_mjx', 'quaterion'],
    install_requires=install_requires,
    python_requires=">=3.8",
    # Note: Entry points would require main() functions in the modules
    # Current files use if __name__ == '__main__' pattern
    # entry_points={
    #     'console_scripts': [
    #         'forestnav-sim=avoidance_v1:main',
    #         'forestnav-mjx=avoidance_v1_mjx:main',
    #     ],
    # },
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