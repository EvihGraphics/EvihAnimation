import mujoco
m = mujoco.MjModel.from_xml_path('/root/Project/GR00T-WholeBodyControl/motionbricks/assets/skeletons/g1/scene_29dof.xml')
print(type(m.cam_user))
try:
    print(len(m.cam_user))
except Exception as e:
    print(e)
