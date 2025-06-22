#!/usr/bin/env python3
"""
Quaternion MCP Server - HTTP Version

Provides quaternion utilities with human-readable explanations.
Just quaternions - nothing else!
Serves over HTTP for Claude Desktop integration.
"""

import numpy as np
from typing import List, Any, Dict
from flask import Flask, request, jsonify
import json

# Try to import MuJoCo for better quaternion operations
try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

app = Flask(__name__)


# ============================================================================
# CORE QUATERNION FUNCTIONS
# ============================================================================

def create_rotation_quat(axis, angle):
    """Create rotation quaternion from axis and angle"""
    if MUJOCO_AVAILABLE:
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array(axis, dtype=float), np.radians(angle))
        return quat
    else:
        # Fallback implementation
        angle_rad = np.radians(angle)
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        w = np.cos(angle_rad / 2)
        x, y, z = np.sin(angle_rad / 2) * axis
        return np.array([w, x, y, z])


def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    if MUJOCO_AVAILABLE:
        result = np.zeros(4)
        mujoco.mju_mulQuat(result, q1, q2)
        return result
    else:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])


def describe_quaternion(q):
    """Convert quaternion to human-readable description"""
    w, x, y, z = q

    # Check for identity
    if abs(w - 1) < 1e-3 and abs(x) < 1e-3 and abs(y) < 1e-3 and abs(z) < 1e-3:
        return "No rotation (identity)"

    # Calculate angle and axis
    angle = 2 * np.arccos(abs(w))
    angle_deg = np.degrees(angle)

    if angle < 1e-3:
        return "No rotation (identity)"

    # Get rotation axis
    vector_mag = np.sqrt(x * x + y * y + z * z)
    if vector_mag < 1e-3:
        return "No rotation (identity)"

    ax, ay, az = x / vector_mag, y / vector_mag, z / vector_mag

    # Identify primary axis
    if abs(ax) > 0.9:
        axis_name = "X-axis"
    elif abs(ay) > 0.9:
        axis_name = "Y-axis"
    elif abs(az) > 0.9:
        axis_name = "Z-axis"
    else:
        axis_name = f"[{ax:.2f}, {ay:.2f}, {az:.2f}]"

    return f"{angle_deg:.1f}° around {axis_name}"


def quat_string(q):
    """Format quaternion as clean string"""
    return f"[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]"


# ============================================================================
# MCP HTTP ENDPOINTS
# ============================================================================

@app.route('/mcp/info', methods=['GET'])
def mcp_info():
    """Return MCP server information"""
    return jsonify({
        "name": "Quaternion Server",
        "version": "1.0.0",
        "description": "Provides quaternion utilities with human-readable explanations",
        "mujoco_available": MUJOCO_AVAILABLE
    })


@app.route('/mcp/tools', methods=['GET'])
def list_tools():
    """List available MCP tools"""
    tools = [
        {
            "name": "explain_quaternion",
            "description": "Explain what a quaternion represents in human terms",
            "parameters": {
                "type": "object",
                "properties": {
                    "quaternion": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "[w, x, y, z] quaternion components"
                    }
                },
                "required": ["quaternion"]
            }
        },
        {
            "name": "create_rotation",
            "description": "Create a rotation quaternion from axis and angle",
            "parameters": {
                "type": "object",
                "properties": {
                    "axis": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Rotation axis [x, y, z]"
                    },
                    "angle": {
                        "type": "number",
                        "description": "Angle in degrees"
                    }
                },
                "required": ["axis", "angle"]
            }
        },
        {
            "name": "multiply_quaternions",
            "description": "Multiply two quaternions (composition of rotations)",
            "parameters": {
                "type": "object",
                "properties": {
                    "q1": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "First quaternion [w, x, y, z]"
                    },
                    "q2": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Second quaternion [w, x, y, z]"
                    }
                },
                "required": ["q1", "q2"]
            }
        },
        {
            "name": "basic_rotations",
            "description": "Create basic rotations around X, Y, or Z axes",
            "parameters": {
                "type": "object",
                "properties": {
                    "axis": {
                        "type": "string",
                        "description": "\"x\", \"y\", or \"z\""
                    },
                    "angle": {
                        "type": "number",
                        "description": "Angle in degrees"
                    }
                },
                "required": ["axis", "angle"]
            }
        },
        {
            "name": "quaternion_inverse",
            "description": "Get the inverse (conjugate) of a quaternion",
            "parameters": {
                "type": "object",
                "properties": {
                    "quaternion": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "[w, x, y, z] quaternion components"
                    }
                },
                "required": ["quaternion"]
            }
        },
        {
            "name": "quaternion_examples",
            "description": "Get common quaternion examples with explanations",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]
    return jsonify({"tools": tools})


@app.route('/mcp/call', methods=['POST'])
def call_tool():
    """Execute an MCP tool"""
    try:
        data = request.json
        tool_name = data.get('tool')
        arguments = data.get('arguments', {})

        if tool_name == 'explain_quaternion':
            result = explain_quaternion_impl(arguments.get('quaternion', []))
        elif tool_name == 'create_rotation':
            result = create_rotation_impl(arguments.get('axis', []), arguments.get('angle', 0))
        elif tool_name == 'multiply_quaternions':
            result = multiply_quaternions_impl(arguments.get('q1', []), arguments.get('q2', []))
        elif tool_name == 'basic_rotations':
            result = basic_rotations_impl(arguments.get('axis', ''), arguments.get('angle', 0))
        elif tool_name == 'quaternion_inverse':
            result = quaternion_inverse_impl(arguments.get('quaternion', []))
        elif tool_name == 'quaternion_examples':
            result = quaternion_examples_impl()
        else:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 400

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def explain_quaternion_impl(quaternion: List[float]) -> str:
    """Explain what a quaternion represents in human terms"""
    if len(quaternion) != 4:
        return "Error: Quaternion must have 4 components [w, x, y, z]"

    q = np.array(quaternion)
    mag = np.linalg.norm(q)
    description = describe_quaternion(q)

    result = f"Quaternion {quat_string(q)}:\n"
    result += f"Represents: {description}\n"
    result += f"Magnitude: {mag:.6f} {'✓' if abs(mag - 1.0) < 1e-6 else '⚠️ Not normalized'}"

    return result


def create_rotation_impl(axis: List[float], angle: float) -> str:
    """Create a rotation quaternion from axis and angle"""
    if len(axis) != 3:
        return "Error: Axis must have 3 components [x, y, z]"

    q = create_rotation_quat(axis, angle)
    description = describe_quaternion(q)

    return f"Rotation: {angle}° around {axis}\nQuaternion: {quat_string(q)}\nRepresents: {description}"


def multiply_quaternions_impl(q1: List[float], q2: List[float]) -> str:
    """Multiply two quaternions (composition of rotations)"""
    if len(q1) != 4 or len(q2) != 4:
        return "Error: Both quaternions must have 4 components"

    result = quaternion_multiply(np.array(q1), np.array(q2))

    return f"Quaternion multiplication:\n{quat_string(q1)} × {quat_string(q2)} = {quat_string(result)}\nResult represents: {describe_quaternion(result)}"


def basic_rotations_impl(axis: str, angle: float) -> str:
    """Create basic rotations around X, Y, or Z axes"""
    axis_map = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}

    if axis.lower() not in axis_map:
        return "Error: Axis must be 'x', 'y', or 'z'"

    q = create_rotation_quat(axis_map[axis.lower()], angle)
    description = describe_quaternion(q)

    return f"{angle}° rotation around {axis.upper()}-axis:\nQuaternion: {quat_string(q)}\nRepresents: {description}"


def quaternion_inverse_impl(quaternion: List[float]) -> str:
    """Get the inverse (conjugate) of a quaternion"""
    if len(quaternion) != 4:
        return "Error: Quaternion must have 4 components"

    q = np.array(quaternion)
    inverse = np.array([q[0], -q[1], -q[2], -q[3]])  # Conjugate for unit quaternions

    return f"Original: {quat_string(q)}\nInverse: {quat_string(inverse)}\nInverse represents: {describe_quaternion(inverse)}"


def quaternion_examples_impl() -> str:
    """Get common quaternion examples with explanations"""
    examples = [
        ([1, 0, 0, 0], "Identity - no rotation"),
        ([0.707, 0.707, 0, 0], "90° pitch up (X-axis)"),
        ([0.707, 0, 0.707, 0], "90° roll right (Y-axis)"),
        ([0.707, 0, 0, 0.707], "90° yaw right (Z-axis)"),
        ([0, 1, 0, 0], "180° pitch (X-axis)"),
        ([0, 0, 1, 0], "180° roll (Y-axis)"),
        ([0, 0, 0, 1], "180° yaw (Z-axis)"),
        ([0.924, 0.383, 0, 0], "45° pitch up (X-axis)"),
        ([0.5, 0.5, 0.5, 0.5], "120° around [1,1,1] diagonal")
    ]

    result = "Common Quaternion Examples:\n\n"
    for q, desc in examples:
        result += f"{quat_string(q)} → {desc}\n"

    result += f"\nMuJoCo available: {MUJOCO_AVAILABLE}"
    return result


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "mujoco": MUJOCO_AVAILABLE})


if __name__ == "__main__":
    print("🔄 Quaternion MCP Server (HTTP)")
    print(f"MuJoCo: {'✓' if MUJOCO_AVAILABLE else '✗'}")
    print("Starting HTTP server on http://localhost:5000")
    print("Ready to serve quaternion tools!")

    # Run the Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)