import os
import sys
import numpy as np
import mujoco

import platform
if platform.system() == 'Windows':
    MOTIONBRICKS_ROOT = r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks'
else:
    MOTIONBRICKS_ROOT = '/root/Project/GR00T-WholeBodyControl/motionbricks'

model_path = f"{MOTIONBRICKS_ROOT}/assets/skeletons/g1/scene_29dof.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

qpos_history = np.load("interactive_qpos.npy")
num_frames = qpos_history.shape[0]

geom_pos = []
geom_rot = []
cam_pos = []
cam_lookat = []

# T_m_to_e: (x, y, z) -> (y, z, x).
T_m_to_e = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
], dtype=np.float32)

for i in range(num_frames):
    d.qpos[:] = qpos_history[i]
    mujoco.mj_forward(m, d)
    
    frame_pos = d.geom_xpos.copy()
    frame_rot = d.geom_xmat.reshape(-1, 3, 3).copy()
    
    # Chase camera logic
    pelvis_pos_m = d.subtree_com[1]
    pelvis_pos_e = T_m_to_e @ pelvis_pos_m
    
    c_target = pelvis_pos_e
    c_pos = pelvis_pos_e + np.array([3.0, 1.5, 0.0])
    
    geom_pos.append(frame_pos)
    geom_rot.append(frame_rot)
    cam_pos.append(c_pos)
    cam_lookat.append(c_target)
    
geom_pos = np.array(geom_pos)
geom_rot = np.array(geom_rot)
cam_pos = np.array(cam_pos)
cam_lookat = np.array(cam_lookat)

# We need to transform from Mujoco space to Evih space
geom_pos_e = np.einsum('ij,ntj->nti', T_m_to_e, geom_pos)
geom_rot_e = np.einsum('ij,ntjk,kl->ntil', T_m_to_e, geom_rot, T_m_to_e.T)

np.save("interactive_geom_pos.npy", geom_pos_e)
np.save("interactive_geom_rot.npy", geom_rot_e)
np.save("interactive_cam_pos.npy", cam_pos)
np.save("interactive_cam_lookat.npy", cam_lookat)

print("Exported geom arrays for interactive trace.")
