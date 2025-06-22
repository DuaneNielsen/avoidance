import mujoco
import glfw
import numpy as np



def draw_overlay(ctx, vp, line1, line2):
    """
    Draws an overlay with two lines of text on the MuJoCo rendering viewport.

    Parameters:
        ctx (mujoco.MjrContext): The rendering context.
        vp (mujoco.MjrRect): The viewport rectangle specifying where to draw.
        line1 (str): The first line of text (displayed on top).
        line2 (str): The second line of text (displayed below the first).
    """
    mujoco.mjr_overlay(
        mujoco.mjtFontScale.mjFONTSCALE_150,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        vp,
        line1,
        line2,
        ctx
    )



# Initialize GLFW and window
glfw.init()
window = glfw.create_window(800, 600, "MuJoCo UI Demo", None, None)
glfw.make_context_current(window)

# Load a simple model
model = mujoco.MjModel.from_xml_string("""
<mujoco><worldbody>
  <body pos="0 0 1">
    <geom type="sphere" size="0.1"/>
  </body>
</worldbody></mujoco>""")
data = mujoco.MjData(model)

# Setup rendering context
ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
scene = mujoco.MjvScene(model, 1000)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
pert = mujoco.MjvPerturb()

# Render loop
while not glfw.window_should_close(window):
    mujoco.mj_step(model, data)

    # Update and render scene
    w, h = glfw.get_framebuffer_size(window)
    vp = mujoco.MjrRect(0, 0, w, h)
    mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(vp, scene, ctx)

    # Draw overlay text
    val = np.sin(data.time) * 5
    overlay1 = f"Time: {data.time:.2f}"
    overlay2 = f"Value: {val:.3f}"
    draw_overlay(ctx, vp, overlay1, overlay2)

    glfw.swap_buffers(window)
    glfw.poll_events()

# Cleanup
ctx.free()
glfw.terminate()
