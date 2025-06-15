import numpy as np
import mujoco
import mujoco.viewer

xml = """
<mujoco model="heightfield_hills">
  <asset>
    <hfield name="terrain" nrow="64" ncol="64" size="4 4 1.0 0.1"/>
    <texture name="grid" type="2d" builtin="checker" 
             width="512" height="512" rgb2="0.2 0.4 0.2" rgb1="0.8 1.0 0.8"/>
    <material name="terrain_mat" texture="grid" texrepeat="8 8" 
              texuniform="true" reflectance="0.3"/>
    <material name="ball_mat" rgba="1 0 0 1"/>
  </asset>

  <worldbody>
    <light name="top" pos="0 0 8" dir="0 0 -1"/>
    
    <!-- Heightfield terrain -->
    <geom name="heightfield" type="hfield" hfield="terrain" material="terrain_mat"/>
    
    <!-- Test ball to show terrain interaction -->
    <body name="ball" pos="0 0 2">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.1" material="ball_mat"/>
    </body>
  </worldbody>
</mujoco>
"""

def generate_hill_terrain(nrow, ncol, hills_x=3, hills_y=3, hill_height=0.8, hill_radius=0.3):
    """
    Generate heightfield data with a grid of circular hills.

    Args:
        nrow, ncol: Heightfield dimensions
        hills_x, hills_y: Number of hills in each direction
        hill_height: Maximum height of hills (0-1 range)
        hill_radius: Radius of hills as fraction of spacing
    """
    terrain = np.zeros((nrow, ncol))

    # Calculate hill positions
    spacing_x = nrow / (hills_x + 1)
    spacing_y = ncol / (hills_y + 1)

    for i in range(hills_x):
        for j in range(hills_y):
            # Hill center positions
            center_x = (i + 1) * spacing_x
            center_y = (j + 1) * spacing_y

            # Create circular hill
            for x in range(nrow):
                for y in range(ncol):
                    # Distance from hill center
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)

                    # Hill radius in grid units
                    radius = hill_radius * min(spacing_x, spacing_y)

                    # Smooth circular hill (cosine falloff)
                    if dist < radius:
                        height = hill_height * (np.cos(np.pi * dist / radius) + 1) / 2
                        terrain[x, y] = max(terrain[x, y], height)

    print(f"Generated {hills_x}x{hills_y} hills")
    print(f"Terrain height range: {terrain.min():.3f} to {terrain.max():.3f}")
    print(f"Hill spacing: {spacing_x:.1f} x {spacing_y:.1f} grid units")

    return terrain.flatten()

def main():
    # Create model
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Generate terrain with 3x3 grid of hills
    nrow, ncol = 64, 64
    terrain_data = generate_hill_terrain(
        nrow=nrow,
        ncol=ncol,
        hills_x=3,      # 3 hills in X direction
        hills_y=3,      # 3 hills in Y direction
        hill_height=0.8, # 80% of max height
        hill_radius=0.4  # 40% of spacing
    )

    # Set heightfield data
    model.hfield_data[:] = terrain_data

    # Launch 3D viewer
    print("\nLaunching 3D viewer...")
    print("- Ball will fall and roll on the terrain")
    print("- Use mouse to navigate camera")
    print("- Red ball shows physics interaction with hills")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()