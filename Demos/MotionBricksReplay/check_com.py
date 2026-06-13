import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\scene_29dof.xml')
d = mujoco.MjData(m)
q = np.load(r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\output\motionbricks_lite\exports\demo_run_01\qpos.npy')
d.qpos[:] = q[50]
mujoco.mj_forward(m, d)
print('COM before:', d.subtree_com[1])
for i in range(m.ngeom):
    if m.geom_group[i] != 1:
        m.geom_pos[i] = [0, 0, 1000]
mujoco.mj_forward(m, d)
print('COM after:', d.subtree_com[1])
