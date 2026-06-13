import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\scene_29dof.xml')
d = mujoco.MjData(m)
q = np.load(r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\output\motionbricks_lite\exports\demo_run_01\qpos.npy')
d.qpos[:] = q[50]
mujoco.mj_forward(m, d)
pos_e = np.load('evih_geom_pos.npy')
cam_pos = np.array([-2.38863349,  1.6768198 ,  0.6791476])
cam_lookat = np.array([0.43044427, 0.65075941, 0.71314758])
cam_dir = cam_lookat - cam_pos
cam_dir /= np.linalg.norm(cam_dir)
cam_right = np.cross(cam_dir, np.array([0,1,0]))
cam_right /= np.linalg.norm(cam_right)
cam_up = np.cross(cam_right, cam_dir)
max_diff = 0
for i in range(m.ngeom):
    if m.geom_group[i] == 1:
        # Convert Evih position back to MuJoCo local relative frame
        # Evih: x=mujoco_y, y=mujoco_z, z=mujoco_x
        # So mujoco_x = evih_z, mujoco_y = evih_x, mujoco_z = evih_y
        mujoco_pos_from_evih = np.array([pos_e[50, i, 2], pos_e[50, i, 0], pos_e[50, i, 1]])
        diff = np.linalg.norm(mujoco_pos_from_evih - d.geom_xpos[i])
        if diff > max_diff:
            max_diff = diff
            max_i = i
print('Max position diff:', max_diff, 'at geom:', max_i)
