#!/usr/bin/env python3
"""
Script to automatically detect dependencies from Python files and generate requirements.txt
"""
import ast
import os
import re
import sys
import importlib.util
from typing import List, Set, Union


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
    target_files = ['avoidance_v1.py', 'avoidance_v1_mjx.py', 'train_ppo.py', 'forestnav_env.py', 'forestnav_xml.py', 'train_jax_ppo.py']
    all_imports = set()
    
    for filepath in target_files:
        if os.path.exists(filepath):
            file_imports = get_imports_from_file(filepath)
            all_imports.update(file_imports)
            print(f"Imports from {filepath}: {sorted(file_imports)}")
    
    # Filter out standard library modules and local imports
    def is_stdlib_module(module_name):
        """Check if a module is part of the Python standard library."""
        if module_name in sys.builtin_module_names:
            return True
        
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                return False
            
            # Standard library modules are typically in the stdlib path
            if spec.origin and 'site-packages' not in spec.origin:
                return True
                
        except (ImportError, ValueError, ModuleNotFoundError):
            pass
        
        return False
    
    # Remove standard library and local modules
    local_modules = {'avoidance_v1', 'quaterion', 'forestnav_xml', 'forestnav_env'}
    external_imports = {
        module for module in all_imports 
        if not is_stdlib_module(module) and module not in local_modules
    }
    
    return sorted(external_imports)


def generate_requirements():
    """Generate requirements.txt file with detected dependencies."""
    detected_deps = get_all_dependencies()
    print(f"Detected external dependencies: {detected_deps}")
    
    # Map only the weird import names to PyPI package names
    package_mapping = {
        'cv2': 'opencv-python==4.11.0.86',
        'jax': 'jax[cpu]',  # or jax[cuda] for GPU support
        'orbax': 'orbax-checkpoint',
        'ml_collections': 'ml-collections',
        'absl': 'absl-py',
        'mujoco_playground': 'playground',
    }
    
    # Convert detected imports to proper package names
    requirements = []
    for dep in detected_deps:
        if dep in package_mapping:
            requirements.append(package_mapping[dep])
        else:
            requirements.append(dep)
    
    # Write requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write("# Auto-generated requirements file\n")
        f.write("# Run: python generate_requirements.py to update\n\n")
        for req in sorted(requirements):
            f.write(f"{req}\n")
    
    print(f"Generated requirements.txt with {len(requirements)} dependencies")
    return requirements


if __name__ == '__main__':
    generate_requirements()