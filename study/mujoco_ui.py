import mujoco
import numpy as np
from mujoco import viewer


def draw_overlay(render_state, line1, line2):
    """
    Draws an overlay with two lines of text in the MuJoCo viewer.

    Parameters:
        render_state (mujoco.viewer.RenderState): The viewer's render state.
        line1 (str): The first line of text (displayed on top).
        line2 (str): The second line of text (displayed below the first).
    """
    mujoco.mjr_overlay(
        mujoco.mjtFontScale.mjFONTSCALE_150,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        render_state.viewport,
        line1,
        line2,
        render_state.context
    )


# Load a simple model
model = mujoco.MjModel.from_xml_string("""
<mujoco><worldbody>
  <body pos="0 0 1">
    <geom type="sphere" size="0.1"/>
  </body>
</worldbody></mujoco>""")
data = mujoco.MjData(model)

# Launch viewer
with viewer.launch_passive(model, data) as viewer_inst:
    while viewer_inst.is_running():
        mujoco.mj_step(model, data)

        # Draw overlay manually during each frame
        val = np.sin(data.time) * 5
        overlay1 = f"Time: {data.time:.2f}"
        overlay2 = f"Value: {val:.3f}"
        draw_overlay(viewer_inst.render_state, overlay1, overlay2)
