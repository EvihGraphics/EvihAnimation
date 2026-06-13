import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\scene_29dof.xml')
d = mujoco.MjData(m)
q = np.load(r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\output\motionbricks_lite\exports\demo_run_01\qpos.npy')
d.qpos[:] = q[50]
mujoco.mj_forward(m, d)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
mujoco.mjv_defaultFreeCamera(m, cam)
cam.distance = 3.0
cam.elevation = -20
cam.azimuth = 90
cam.lookat[:] = d.subtree_com[1]
renderer = mujoco.Renderer(m, 480, 640)
m.vis.global_.fovy = 34.0

for i in range(m.ngeom):
    if m.geom_group[i] != 1:
        m.geom_pos[i] = [0, 0, 1000]

renderer.update_scene(d, camera=cam)
renderer.enable_segmentation_rendering()
seg = renderer.render()
mask_m = (seg[:,:,0] > 0)
print('Unique segment IDs:', np.unique(seg[:,:,0]))
print('Sum of pixels > 0:', np.sum(mask_m))

import cv2
mask_e_img = cv2.imread(r'D:\AnimationTech-learning\EvihAnimation-motionbricks-replay\Demos\MotionBricksReplay\frames\frame_0051.png')
mask_e = ((mask_e_img[:,:,0]<50) & (mask_e_img[:,:,1]<50) & (mask_e_img[:,:,2]<50))

iou = np.sum(mask_m & mask_e) / np.sum(mask_m | mask_e)
print('IoU:', iou)

overlay = np.zeros((480, 640, 3), dtype=np.uint8)
overlay[mask_m] = [0, 0, 255] # Red for MuJoCo
overlay[mask_e] = [0, 255, 0] # Green for Evih
overlap = (mask_m) & (mask_e)
overlay[overlap] = [0, 255, 255] # Yellow for both
cv2.imwrite('best_fovy_overlay_clean.png', overlay)

rm, cm=np.where(mask_m); re, ce=np.where(mask_e)
print(f'MuJoCo BB: Y=[{rm.min()}, {rm.max()}], X=[{cm.min()}, {cm.max()}], W={cm.max()-cm.min()}, H={rm.max()-rm.min()}, COM: x={cm.mean():.1f}, y={rm.mean():.1f}')
print(f'Evih BB: Y=[{re.min()}, {re.max()}], X=[{ce.min()}, {ce.max()}], W={ce.max()-ce.min()}, H={re.max()-re.min()}, COM: x={ce.mean():.1f}, y={re.mean():.1f}')
