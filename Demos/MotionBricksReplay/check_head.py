import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\scene_29dof.xml')
d = mujoco.MjData(m)
q = np.load(r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\output\motionbricks_lite\exports\demo_run_01\qpos.npy')
d.qpos[:] = q[50]
mujoco.mj_forward(m, d)
print('MuJoCo head world pos:', d.geom_xpos[40])
pos_e = np.load('evih_geom_pos.npy')
print('Evih head world pos:', pos_e[50, 40])
cam_lookat_e = np.load('evih_cam_lookat.npy')
print('Evih head rel lookat:', pos_e[50, 40] - cam_lookat_e[50])
print('MuJoCo head rel lookat:', d.geom_xpos[40] - d.subtree_com[1])
