"""
Enhanced Heightfield Rangefinder Test with Complex Hilly Terrain
"""
import numpy as np
import mujoco
import mujoco.viewer as viewer

xml_template = """
<mujoco>
    <asset>
        <!-- Larger heightfield for more complex terrain -->
        <hfield name="terrain" nrow="128" ncol="128" size="8 8 2.0 0.1"/>
        <texture name="grid" type="2d" builtin="checker" 
                 width="512" height="512" rgb2="0.1 0.3 0.1" rgb1="0.3 0.7 0.3"/>
        <material name="terrain_material" texture="grid" texrepeat="8 8" 
                  texuniform="true" reflectance="0.2"/>
        <material name="box_material" rgba="1 0 0 1"/>
        <material name="sensor_material" rgba="0 1 0 1"/>
    </asset>

    <visual>
        <scale forcewidth="0.1" contactwidth="0.3" contactheight="0.1"/>
        <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/>
    </visual>

    <worldbody>
        <light name="sun" pos="0 0 10" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
        <light name="ambient" directional="false" pos="0 0 5" diffuse="0.3 0.3 0.3"/>

        <!-- Large heightfield terrain -->
        <geom name="heightfield" type="hfield" hfield="terrain" material="terrain_material"/>

        <!-- Test objects scattered across terrain -->
        <geom name="test_box1" type="box" size="0.3 0.3 0.5" pos="2.0 1.0 1.0" material="box_material"/>
        <geom name="test_box2" type="box" size="0.2 0.2 0.8" pos="-1.5 2.5 1.0" material="box_material"/>
        <geom name="test_sphere" type="sphere" size="0.4" pos="1.5 -2.0 1.0" rgba="0 0 1 1"/>

        <!-- Mobile sensor platform -->
        <body name="sensor_platform" pos="0 0 2.0">
            <freejoint name="platform_joint"/>
            <geom name="platform" type="cylinder" size="0.15 0.08" material="sensor_material"/>

            <!-- All sensors now pointing downward -->
            <!-- Center downward sensor -->
            <site name="ray_down_center" pos="0 0 0" quat="0.707 0.707 0 0" size="0.03" rgba="1 0 0 1"/>
            
            <!-- Downward sensors with slight angles -->
            <site name="ray_down_forward" pos="0 0 0" quat="0.612 0.612 0.354 0" size="0.02" rgba="1 0.5 0 1"/>
            <site name="ray_down_back" pos="0 0 0" quat="0.612 0.612 -0.354 0" size="0.02" rgba="1 0.5 0 1"/>
            <site name="ray_down_left" pos="0 0 0" quat="0.612 0.612 0 0.354" size="0.02" rgba="0 1 0.5 1"/>
            <site name="ray_down_right" pos="0 0 0" quat="0.612 0.612 0 -0.354" size="0.02" rgba="0 1 0.5 1"/>
            
            <!-- Additional downward sensors at more angles -->
            <site name="ray_down_fl" pos="0 0 0" quat="0.574 0.574 0.250 0.250" size="0.02" rgba="0 0 1 1"/>
            <site name="ray_down_fr" pos="0 0 0" quat="0.574 0.574 -0.250 0.250" size="0.02" rgba="0 0 1 1"/>
            <site name="ray_down_bl" pos="0 0 0" quat="0.574 0.574 0.250 -0.250" size="0.02" rgba="1 1 0 1"/>
        </body>
    </worldbody>

    <sensor>
        <rangefinder name="down_center_sensor" site="ray_down_center"/>
        <rangefinder name="down_forward_sensor" site="ray_down_forward"/>
        <rangefinder name="down_back_sensor" site="ray_down_back"/>
        <rangefinder name="down_left_sensor" site="ray_down_left"/>
        <rangefinder name="down_right_sensor" site="ray_down_right"/>
        <rangefinder name="down_fl_sensor" site="ray_down_fl"/>
        <rangefinder name="down_fr_sensor" site="ray_down_fr"/>
        <rangefinder name="down_bl_sensor" site="ray_down_bl"/>
    </sensor>

    <actuator>
        <!-- Control for moving the sensor platform -->
        <motor name="move_x" joint="platform_joint" gear="1 0 0 0 0 0"/>
        <motor name="move_y" joint="platform_joint" gear="0 1 0 0 0 0"/>
        <motor name="move_z" joint="platform_joint" gear="0 0 1 0 0 0"/>
    </actuator>
</mujoco>
"""


def generate_complex_terrain(nrow, ncol):
    """Generate complex hilly terrain with multiple features"""
    print(f"Generating {nrow}x{ncol} complex terrain...")

    # Initialize terrain
    terrain = np.zeros((nrow, ncol))

    # Create coordinate grids
    x = np.linspace(-4, 4, ncol)
    y = np.linspace(-4, 4, nrow)
    X, Y = np.meshgrid(x, y)

    # Base rolling hills using sine waves
    terrain += 0.3 * np.sin(X * 0.8) * np.cos(Y * 0.6)
    terrain += 0.2 * np.sin(X * 1.5) * np.sin(Y * 1.2)

    # Add several prominent peaks
    peaks = [
        {'center': (1.5, 2.0), 'height': 1.2, 'width': 1.0},
        {'center': (-2.0, -1.0), 'height': 1.0, 'width': 1.2},
        {'center': (0.5, -2.5), 'height': 0.8, 'width': 0.8},
        {'center': (-1.5, 1.5), 'height': 0.9, 'width': 1.1},
        {'center': (2.5, -0.5), 'height': 0.7, 'width': 0.9},
    ]

    for peak in peaks:
        cx, cy = peak['center']
        height = peak['height']
        width = peak['width']

        # Gaussian-like hill
        dist_sq = ((X - cx) ** 2 + (Y - cy) ** 2) / (width ** 2)
        hill = height * np.exp(-dist_sq)
        terrain += hill

    # Add some valleys (negative hills)
    valleys = [
        {'center': (-0.5, 0.5), 'depth': -0.4, 'width': 0.8},
        {'center': (1.0, -1.0), 'depth': -0.3, 'width': 0.6},
    ]

    for valley in valleys:
        cx, cy = valley['center']
        depth = valley['depth']
        width = valley['width']

        dist_sq = ((X - cx) ** 2 + (Y - cy) ** 2) / (width ** 2)
        depression = depth * np.exp(-dist_sq)
        terrain += depression

    # Add fine-scale noise for realistic texture
    noise = np.random.normal(0, 0.05, (nrow, ncol))

    # Smooth the noise
    from scipy.ndimage import gaussian_filter
    noise = gaussian_filter(noise, sigma=1.0)
    terrain += noise

    # Add some ridges
    ridge1 = 0.3 * np.exp(-((X + Y - 1) ** 2) / 0.1) * np.exp(-((X - Y) ** 2) / 2.0)
    ridge2 = 0.25 * np.exp(-((X - Y + 2) ** 2) / 0.15) * np.exp(-((X + Y) ** 2) / 1.5)
    terrain += ridge1 + ridge2

    # Ensure minimum height is 0
    terrain = np.maximum(terrain, 0)

    print(f"Terrain height range: {terrain.min():.3f} to {terrain.max():.3f}")
    print(f"Terrain mean height: {terrain.mean():.3f}")

    return terrain.flatten()


def move_platform_over_terrain(model, data, t):
    """Move the sensor platform in a pattern over the terrain - NO ROLL"""
    # Circular motion with some vertical component
    radius = 2.0
    speed = 0.01  # Keep the slow speed you set

    x = radius * np.cos(speed * t)
    y = radius * np.sin(speed * t) * 0.7  # Elliptical
    z = 2.0 + 0.5 * np.sin(speed * t * 2)  # Vary height

    # Set platform position
    data.qpos[0:3] = [x, y, z]

    # NO ROLL OR PITCH - only slow yaw rotation
    yaw = speed * t * 0.5  # Just slow yaw rotation

    # Convert to quaternion (only yaw)
    data.qpos[6] = np.cos(yaw / 2)  # qw
    data.qpos[3] = 0                # qx (no roll)
    data.qpos[4] = 0                # qy (no pitch)
    data.qpos[5] = np.sin(yaw / 2)  # qz (yaw only)


def print_sensor_readings(data):
    """Print current sensor readings"""
    sensor_names = [
        'down_center_sensor', 'down_forward_sensor', 'down_back_sensor',
        'down_left_sensor', 'down_right_sensor', 'down_fl_sensor',
        'down_fr_sensor', 'down_bl_sensor'
    ]

    print(f"Time: {data.time:6.2f}s - Downward Sensor readings:")
    for i, name in enumerate(sensor_names):
        reading = data.sensordata[i]
        status = 'HIT' if reading > 0 else 'MISS'
        print(f"  {name:18s}: {reading:6.3f}m ({status})")
    print()


def main():
    # Generate complex terrain
    nrow, ncol = 128, 128
    terrain_data = generate_complex_terrain(nrow, ncol)

    # Create model and set terrain dynamically
    model = mujoco.MjModel.from_xml_string(xml_template)
    model.hfield_data[:] = terrain_data  # Dynamic terrain assignment
    data = mujoco.MjData(model)

    print("Complex Hilly Terrain Rangefinder Test")
    print("- Large 128x128 heightfield")
    print("- Multiple hills, valleys, and ridges")
    print("- Mobile sensor platform with 8 DOWNWARD-POINTING rangefinders")
    print("- NO ROLL motion - platform stays level")
    print("- Very slow movement for detailed observation")
    print("- Real-time sensor readings")
    print()

    # Set up visualization
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # Enhance ray visibility
    model.vis.scale.forcewidth = 0.03
    model.vis.scale.contactwidth = 0.1

    # Launch viewer with dynamic motion
    step = 0
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            # Move platform over terrain
            move_platform_over_terrain(model, data, data.time)

            # Step simulation
            mujoco.mj_step(model, data)

            # Print sensor readings every 30 steps
            if step % 30 == 0:
                print_sensor_readings(data)

            # Update visualization
            v.opt.flags = scene_option.flags
            v.sync()

            step += 1


if __name__ == "__main__":
    main()