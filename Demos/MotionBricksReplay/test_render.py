import mujoco
m = mujoco.MjModel.from_xml_path(r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\scene_29dof.xml')
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 480, 640)
print("Has track?", "track" in [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)])
if m.cam_user is not None:
    try:
        r.update_scene(d, camera="track")
        print("Success with track")
    except Exception as e:
        print("Crash:", e)
