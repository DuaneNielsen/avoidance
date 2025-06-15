import numpy as np
import mujoco
import mujoco.viewer as viewer

xml_template = """
<mujoco>
    <asset>
        <hfield name="terrain" nrow="32" ncol="32" size="2 2 1.0 0.1"/>
        <texture name="grid" type="2d" builtin="checker" 
                 width="512" height="512" rgb2="0 0 0" rgb1="1 1 1"/>
        <material name="grid" texture="grid" texrepeat="4 4" 
                  texuniform="true" reflectance="0.3"/>
        <material name="box_material" rgba="1 0 0 1"/>
        <material name="ground_material" rgba="0.5 0.5 0.5 1"/>
        <material name="sensor_material" rgba="0 1 0 1"/>
    </asset>

    <visual>
        <!-- Make rangefinder rays more visible -->
        <scale forcewidth="0.1" contactwidth="0.3" contactheight="0.1"/>
    </visual>

    <worldbody>
        <light name="top" pos="0 0 8"/>

        <!-- Heightfield terrain with hill offset from center -->
        <geom name="heightfield" type="hfield" hfield="terrain" material="grid"/>

        <!-- Test box -->
        <geom name="test_box" type="box" size="0.2 0.2 0.3" pos="1.0 0 0.5" material="box_material"/>

        <!-- Sensor platform at origin -->
        <body name="sensor_platform" pos="0 0 0.3">
            <geom name="platform" type="box" size="0.1 0.1 0.05" material="sensor_material"/>

            <!-- Ray 1: Points down at heightfield (should HIT but MISSES) -->
            <site name="ray_heightfield_hit" pos="0 0 0" quat="0.707 0.707 0 0" size="0.02" rgba="0 1 0 1"/>

            <!-- Ray 2: Points at box (should HIT but MISSES) -->
            <site name="ray_box_hit" pos="0 0 0" quat="0.924 0 0.383 0" size="0.02" rgba="0 0 1 1"/>

            <!-- Ray 3: Points up at sky (should MISS but HITS) -->
            <site name="ray_miss_sky" pos="0 0 0" quat="0.707 -0.707 0 0" size="0.02" rgba="1 0 0 1"/>

            <!-- Ray 4: Points horizontally away (should MISS but HITS) -->
            <site name="ray_miss_horizon" pos="0 0 0" quat="0.707 0 0.707 0" size="0.02" rgba="1 1 0 1"/>
        </body>
    </worldbody>

    <sensor>
        <rangefinder name="heightfield_sensor" site="ray_heightfield_hit"/>
        <rangefinder name="box_sensor" site="ray_box_hit"/>
        <rangefinder name="sky_sensor" site="ray_miss_sky"/>
        <rangefinder name="horizon_sensor" site="ray_miss_horizon"/>
    </sensor>
</mujoco>
"""


def generate_test_terrain(nrow, ncol):
    """Generate test terrain with hill offset from center"""
    terrain = np.zeros((nrow, ncol))

    # Create a hill offset to one side
    hill_center_x, hill_center_y = nrow - 8, ncol // 2
    for i in range(nrow):
        for j in range(ncol):
            dist = np.sqrt((i - hill_center_x) ** 2 + (j - hill_center_y) ** 2)
            if dist < 6:
                terrain[i, j] = 0.8 * (1 - dist / 6)

    print(f"Terrain height range: {terrain.min():.3f} to {terrain.max():.3f}")
    return terrain.flatten()


def main():
    # Generate terrain data
    nrow, ncol = 32, 32
    terrain_data = generate_test_terrain(nrow, ncol)

    # Create model and set terrain
    model = mujoco.MjModel.from_xml_string(xml_template)
    model.hfield_data[:] = terrain_data
    data = mujoco.MjData(model)

    print("Enhanced ray visualization:")
    print("- Larger site markers")
    print("- Enhanced visual scaling")
    print()

    # Set up visualization options
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # Make rays more visible
    model.vis.scale.forcewidth = 0.05
    model.vis.scale.contactwidth = 0.2

    # Step once to get readings
    mujoco.mj_step(model, data)

    # Print detailed results with site positions
    heightfield_reading = data.sensordata[0]
    box_reading = data.sensordata[1]
    sky_reading = data.sensordata[2]
    horizon_reading = data.sensordata[3]

    print("Results:")
    print(f"Heightfield: {heightfield_reading:6.3f}m - {'HIT' if heightfield_reading > 0 else 'MISS'}")
    print(f"Box:         {box_reading:6.3f}m - {'HIT' if box_reading > 0 else 'MISS'}")
    print(f"Sky:         {sky_reading:6.3f}m - {'HIT' if sky_reading > 0 else 'MISS'}")
    print(f"Horizon:     {horizon_reading:6.3f}m - {'HIT' if horizon_reading > 0 else 'MISS'}")

    # Debug site directions
    print("\nSite directions:")
    for i, name in enumerate(['ray_heightfield_hit', 'ray_box_hit', 'ray_miss_sky', 'ray_miss_horizon']):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        site_pos = data.site_xpos[site_id]
        site_mat = data.site_xmat[site_id].reshape(3, 3)
        direction = site_mat[:, 0]  # X-axis is pointing direction
        print(f"{name}: pos={site_pos}, dir={direction}")

    # Launch viewer
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            v.opt.flags = scene_option.flags
            v.sync()


if __name__ == "__main__":
    main()