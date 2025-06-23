#!/usr/bin/env python3
"""
Quaternion MCP Server - FastMCP Version

Provides quaternion utilities with human-readable explanations.
Just quaternions - nothing else!
Uses FastMCP for proper MCP protocol compliance.
"""

import numpy as np
from typing import List
import asyncio

# Try to import MuJoCo for better quaternion operations
try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

# Import FastMCP
try:
    from fastmcp import FastMCP
except ImportError:
    print("Error: FastMCP not installed. Install with: pip install fastmcp")
    exit(1)

# Create the MCP server
mcp = FastMCP("Quaternion Server")


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
# MCP TOOLS
# ============================================================================

@mcp.tool()
def explain_quaternion(quaternion: List[float]) -> str:
    """Explain what a quaternion represents in human terms

    Args:
        quaternion: List of 4 float components [w, x, y, z]

    Returns:
        Human-readable explanation of the quaternion
    """
    if len(quaternion) != 4:
        return "Error: Quaternion must have 4 components [w, x, y, z]"

    q = np.array(quaternion)
    mag = np.linalg.norm(q)
    description = describe_quaternion(q)

    result = f"Quaternion {quat_string(q)}:\n"
    result += f"Represents: {description}\n"
    result += f"Magnitude: {mag:.6f} {'✓' if abs(mag - 1.0) < 1e-6 else '⚠️ Not normalized'}"

    return result


@mcp.tool()
def create_rotation(axis: List[float], angle: float) -> str:
    """Create a rotation quaternion from axis and angle

    Args:
        axis: Rotation axis as [x, y, z]
        angle: Angle in degrees

    Returns:
        Quaternion representation and description
    """
    if len(axis) != 3:
        return "Error: Axis must have 3 components [x, y, z]"

    q = create_rotation_quat(axis, angle)
    description = describe_quaternion(q)

    return f"Rotation: {angle}° around {axis}\nQuaternion: {quat_string(q)}\nRepresents: {description}"


@mcp.tool()
def multiply_quaternions(q1: List[float], q2: List[float]) -> str:
    """Multiply two quaternions (composition of rotations)

    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]

    Returns:
        Result of quaternion multiplication with description
    """
    if len(q1) != 4 or len(q2) != 4:
        return "Error: Both quaternions must have 4 components"

    result = quaternion_multiply(np.array(q1), np.array(q2))

    return f"Quaternion multiplication:\n{quat_string(q1)} × {quat_string(q2)} = {quat_string(result)}\nResult represents: {describe_quaternion(result)}"


@mcp.tool()
def basic_rotations(axis: str, angle: float) -> str:
    """Create basic rotations around X, Y, or Z axes

    Args:
        axis: "x", "y", or "z"
        angle: Angle in degrees

    Returns:
        Quaternion for the basic rotation
    """
    axis_map = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}

    if axis.lower() not in axis_map:
        return "Error: Axis must be 'x', 'y', or 'z'"

    q = create_rotation_quat(axis_map[axis.lower()], angle)
    description = describe_quaternion(q)

    return f"{angle}° rotation around {axis.upper()}-axis:\nQuaternion: {quat_string(q)}\nRepresents: {description}"


@mcp.tool()
def quaternion_inverse(quaternion: List[float]) -> str:
    """Get the inverse (conjugate) of a quaternion

    Args:
        quaternion: Quaternion components [w, x, y, z]

    Returns:
        Inverse quaternion with description
    """
    if len(quaternion) != 4:
        return "Error: Quaternion must have 4 components"

    q = np.array(quaternion)
    inverse = np.array([q[0], -q[1], -q[2], -q[3]])  # Conjugate for unit quaternions

    return f"Original: {quat_string(q)}\nInverse: {quat_string(inverse)}\nInverse represents: {describe_quaternion(inverse)}"


@mcp.tool()
def quaternion_examples() -> str:
    """Get common quaternion examples with explanations

    Returns:
        List of common quaternions and what they represent
    """
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
# RESOURCE (Optional - for server info)
# ============================================================================

@mcp.resource("server://info")
def server_info() -> str:
    """Get server information and capabilities"""
    return f"""Quaternion MCP Server
Version: 1.0.0
Description: Provides quaternion utilities with human-readable explanations
MuJoCo Available: {MUJOCO_AVAILABLE}

Available Tools:
- explain_quaternion: Explain what a quaternion represents
- create_rotation: Create rotation quaternion from axis/angle  
- multiply_quaternions: Multiply two quaternions
- basic_rotations: Create basic X/Y/Z axis rotations
- quaternion_inverse: Get quaternion inverse/conjugate
- quaternion_examples: Common quaternion examples
"""


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    print("Quaternion MCP Server (FastMCP)")
    print(f"MuJoCo: {'OK' if MUJOCO_AVAILABLE else 'Not Available'}")

    # Check if running with HTTP transport argument
    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        print("Starting MCP server with HTTP transport...")
        print("Server will be available at http://localhost:5000/mcp")
        print("Ready to serve quaternion tools via MCP protocol!")

        # Run with HTTP transport
        asyncio.run(mcp.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=5000,
            path="/mcp"
        ))
    else:
        print("Starting MCP server with STDIO transport...")
        print("Ready to serve quaternion tools via MCP protocol!")

        # Run with STDIO transport (default for Claude Desktop)
        asyncio.run(mcp.run(transport="stdio"))