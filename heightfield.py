import numpy as np
import mujoco
import matplotlib.pyplot as plt


def create_dramatic_heightfield():
    """Create a heightfield with more dramatic height variations"""

    # Create heightfield data with more variation
    nrow, ncol = 32, 32
    x = np.linspace(-3, 3, ncol)
    y = np.linspace(-3, 3, nrow)
    X, Y = np.meshgrid(x, y)

    # Create more dramatic terrain features
    heights = (
            0.4 * np.sin(1.5 * X) * np.cos(1.5 * Y) +  # Main waves
            0.3 * np.sin(3 * X + 2 * Y) +  # Secondary waves
            0.2 * np.cos(2 * Y - 1.5 * X) +  # Tertiary waves
            0.3 * np.exp(-((X - 1) ** 2 + (Y + 1) ** 2) / 2) +  # Peak 1
            0.25 * np.exp(-((X + 1.5) ** 2 + (Y - 1.5) ** 2) / 1.5)  # Peak 2
    )

    # Add some noise for roughness
    np.random.seed(42)
    noise = 0.1 * np.random.random((nrow, ncol))
    heights += noise

    # Normalize to 0-1 range (this is important!)
    heights = (heights - heights.min()) / (heights.max() - heights.min())

    # FIXED: Better string formatting with proper precision
    height_values = []
    for i in range(nrow):
        row_values = []
        for j in range(ncol):
            row_values.append(f"{heights[i, j]:.6f}")
        height_values.append(" ".join(row_values))

    # Join all rows with newlines for better readability
    height_str = "\n              ".join(height_values)

    # Create XML with proper indentation
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="dramatic_heightfield_terrain">
  <compiler autolimits="true"/>

  <option timestep="0.01" integrator="RK4"/>

  <asset>
    <hfield name="terrain" nrow="{nrow}" ncol="{ncol}" 
            size="3.0 3.0 2.0 0.1">
              {height_str}
    </hfield>

    <!-- Better textures and materials -->
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" 
             rgb1="0.3 0.7 0.3" rgb2="0.2 0.5 0.2"/>
    <texture name="sky" type="skybox" builtin="gradient" 
             rgb1="0.7 0.9 1.0" rgb2="0.3 0.5 0.9" width="256" height="256"/>

    <material name="terrain_mat" texture="grid" texrepeat="16 16" 
              specular="0.2" shininess="0.1" rgba="0.6 0.8 0.4 1"/>
    <material name="ball_mat" rgba="1 0.3 0.3 1" specular="0.9" shininess="0.6"/>
    <material name="box_mat" rgba="0.3 0.3 1 1" specular="0.7" shininess="0.4"/>
  </asset>

  <worldbody>
    <!-- Better lighting -->
    <light name="sun" pos="4 4 8" dir="-1 -1 -3" diffuse="1.0 1.0 0.9"/>
    <light name="ambient" pos="0 0 10" dir="0 0 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Multiple camera angles -->
    <camera name="overview" pos="5 5 4" xyaxes="-1 1 0 0 0 1"/>
    <camera name="side" pos="5 0 3" xyaxes="0 1 0 0 0 1"/>
    <camera name="top" pos="0 0 8" xyaxes="1 0 0 0 1 0"/>
    <camera name="close" pos="2 2 2" xyaxes="-1 1 0 0 0 1"/>

    <!-- Heightfield terrain -->
    <geom name="terrain" type="hfield" hfield="terrain" material="terrain_mat"/>

    <!-- Dynamic objects at different heights -->
    <body name="red_sphere" pos="0 0 3">
      <freejoint/>
      <geom name="ball1" type="sphere" size="0.15" material="ball_mat"/>
    </body>

    <body name="blue_sphere" pos="1.5 -1.5 3.5">
      <freejoint/>
      <geom name="ball2" type="sphere" size="0.12" material="box_mat"/>
    </body>

    <body name="yellow_box" pos="-1.5 1.5 2.5">
      <freejoint/>
      <geom name="box1" type="box" size="0.1 0.1 0.1" rgba="1 1 0 1"/>
    </body>

    <body name="green_capsule" pos="0.8 1.8 3">
      <freejoint/>
      <geom name="capsule1" type="capsule" size="0.06" fromto="0 0 -0.15 0 0 0.15" rgba="0 1 0 1"/>
    </body>

    <body name="purple_cylinder" pos="-2 -0.5 2.8">
      <freejoint/>
      <geom name="cyl1" type="cylinder" size="0.08 0.12" rgba="0.8 0.2 0.8 1"/>
    </body>

    <!-- Reference markers at terrain level -->
    <body name="marker1" pos="2 2 0">
      <geom name="marker1" type="cylinder" size="0.03 1.0" rgba="1 0 1 0.8" 
            contype="0" conaffinity="0"/>
    </body>

    <body name="marker2" pos="-2 -2 0">
      <geom name="marker2" type="cylinder" size="0.03 1.0" rgba="0 1 1 0.8" 
            contype="0" conaffinity="0"/>
    </body>

    <!-- Corner markers to show terrain bounds -->
    <body name="corner1" pos="3 3 0">
      <geom name="c1" type="box" size="0.05 0.05 0.5" rgba="1 0 0 0.7" 
            contype="0" conaffinity="0"/>
    </body>

    <body name="corner2" pos="-3 -3 0">
      <geom name="c2" type="box" size="0.05 0.05 0.5" rgba="1 0 0 0.7" 
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

    return xml, heights


def create_simple_clean_heightfield():
    """Create a simple, clean heightfield with proper formatting"""

    # Create a simple 16x16 heightfield for easier debugging
    nrow, ncol = 16, 16
    heights = np.zeros((nrow, ncol), dtype=np.float64)

    # Create a simple pattern: pyramid in center
    center_r, center_c = nrow // 2, ncol // 2
    for i in range(nrow):
        for j in range(ncol):
            # Distance from center
            dist_r = abs(i - center_r)
            dist_c = abs(j - center_c)
            max_dist = max(center_r, center_c)
            # Create pyramid that decreases with distance
            height = max(0.0, 1.0 - max(dist_r, dist_c) / max_dist)
            heights[i, j] = height

    # Create properly formatted height string
    height_lines = []
    for i in range(nrow):
        row_str = " ".join(f"{heights[i, j]:.6f}" for j in range(ncol))
        height_lines.append(row_str)

    # Join with proper indentation
    height_str = "\n              ".join(height_lines)

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="simple_clean_heightfield">
  <compiler autolimits="true"/>

  <option timestep="0.01" integrator="RK4"/>

  <asset>
    <hfield name="terrain" nrow="{nrow}" ncol="{ncol}" 
            size="2.0 2.0 1.5 0.1">
              {height_str}
    </hfield>

    <material name="terrain_mat" rgba="0.4 0.7 0.3 1"/>
    <material name="ball_mat" rgba="1 0.2 0.2 1"/>
  </asset>

  <worldbody>
    <light name="sun" pos="3 3 5" dir="-1 -1 -2" diffuse="0.9 0.9 0.8"/>
    <light name="ambient" pos="0 0 8" dir="0 0 -1" diffuse="0.3 0.3 0.3"/>

    <camera name="overview" pos="4 4 3" xyaxes="-1 1 0 0 0 1"/>
    <camera name="side" pos="4 0 2" xyaxes="0 1 0 0 0 1"/>
    <camera name="top" pos="0 0 6" xyaxes="1 0 0 0 1 0"/>

    <!-- Heightfield terrain -->
    <geom name="terrain" type="hfield" hfield="terrain" material="terrain_mat"/>

    <!-- Single test object -->
    <body name="test_ball" pos="0 0 3">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.1" material="ball_mat"/>
    </body>

    <!-- Reference markers -->
    <body name="center_marker" pos="0 0 0">
      <geom name="center" type="cylinder" size="0.02 0.8" rgba="1 1 0 0.8" 
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

    return xml, heights


try:
    # Create simple clean version first
    simple_xml, simple_heights = create_simple_clean_heightfield()

    with open('simple_clean_heightfield.xml', 'w') as f:
        f.write(simple_xml)

    print("✅ Simple clean heightfield written to: simple_clean_heightfield.xml")

    # Create dramatic version with fixed formatting
    dramatic_xml, dramatic_heights = create_dramatic_heightfield()

    with open('dramatic_heightfield_fixed.xml', 'w') as f:
        f.write(dramatic_xml)

    print("✅ Dramatic heightfield (fixed) written to: dramatic_heightfield_fixed.xml")

    # Test loading both
    simple_model = mujoco.MjModel.from_xml_string(simple_xml)
    dramatic_model = mujoco.MjModel.from_xml_string(dramatic_xml)

    print(f"✅ Simple model: {simple_model.ngeom} geoms")
    print(f"✅ Dramatic model: {dramatic_model.ngeom} geoms")

    # Show the height data for verification
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(simple_heights, cmap='terrain', origin='lower')
    plt.colorbar(label='Height')
    plt.title('Simple Clean Heightfield (16x16)')

    plt.subplot(1, 2, 2)
    plt.imshow(dramatic_heights, cmap='terrain', origin='lower')
    plt.colorbar(label='Height')
    plt.title('Dramatic Heightfield (32x32)')

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("FIXED FORMATTING ISSUES:")
    print("=" * 60)
    print("1. Proper floating point precision (6 decimal places)")
    print("2. Organized height data into rows")
    print("3. Proper XML indentation")
    print("4. Cleaner structure")
    print("\nTO TEST:")
    print("simulate simple_clean_heightfield.xml")
    print("simulate dramatic_heightfield_fixed.xml")

    # Print a sample of the height data to verify formatting
    print("\nSample height data formatting:")
    print("First 3 rows of simple heightfield:")
    for i in range(3):
        row_data = " ".join(f"{simple_heights[i, j]:.6f}" for j in range(min(8, simple_heights.shape[1])))
        print(f"Row {i}: {row_data}...")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()