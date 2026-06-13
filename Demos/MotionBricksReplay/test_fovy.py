import mujoco, numpy as np, cv2

m = mujoco.MjModel.from_xml_path(r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\scene_29dof.xml')
d = mujoco.MjData(m)
q = np.load(r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\output\motionbricks_lite\exports\demo_run_01\qpos.npy')
d.qpos[:] = q[50]

mujoco.mj_forward(m, d)

# Hide collision geoms for clean visual mask by moving them away
for i in range(m.ngeom):
    if m.geom_group[i] != 1:
        m.geom_pos[i] = [0, 0, 1000]

mujoco.mj_forward(m, d)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
mujoco.mjv_defaultFreeCamera(m, cam)
cam.distance = 3.0
cam.elevation = -20
cam.azimuth = 90
cam.lookat[:] = d.subtree_com[1]

renderer = mujoco.Renderer(m, 480, 640)

mask_eimg = cv2.imread(r'D:\AnimationTech-learning\EvihAnimation-motionbricks-replay\Demos\MotionBricksReplay\frames\frame_0051.png')
mask_e = ((mask_eimg[:,:,0]<50) & (mask_eimg[:,:,1]<50) & (mask_eimg[:,:,2]<50))

best_iou = 0
best_fovy = 0

for fovy in np.arange(10.0, 75.0, 0.5):
    m.vis.global_.fovy = fovy
    renderer.update_scene(d, camera=cam)
    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    mask_m = (seg[:, :, 0] > 0)
    
    iou = np.sum(mask_m & mask_e) / np.sum(mask_m | mask_e)
    if iou > best_iou:
        best_iou = iou
        best_fovy = fovy
        
        overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay[mask_m > 0] = [0, 0, 255] # Red for MuJoCo
        overlay[mask_e > 0] = [0, 255, 0] # Green for Evih
        overlap = (mask_m > 0) & (mask_e > 0)
        overlay[overlap] = [0, 255, 255] # Yellow for both
        best_overlay = overlay

print(f"Best FOVY: {best_fovy}, IoU: {best_iou}")
cv2.imwrite('best_fovy_overlay.png', best_overlay)
cv2.imwrite('mask_m_test.png', (mask_m * 255).astype(np.uint8))
cv2.imwrite('mask_e_test.png', (mask_e * 255).astype(np.uint8))
