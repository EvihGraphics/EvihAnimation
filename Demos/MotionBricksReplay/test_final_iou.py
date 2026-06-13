import mujoco, numpy as np, cv2
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
vopt = mujoco.MjvOption()
vopt.geomgroup[0] = 0
vopt.geomgroup[1] = 1

for frame_idx in [10, 50, 100, 200, 290]:
    d.qpos[:] = q[frame_idx]
    mujoco.mj_forward(m, d)

    # Set camera to match Evih
    cam.distance = 3.0
    cam.elevation = -20
    cam.azimuth = 90
    cam.lookat[:] = d.subtree_com[1]  # Track pelvis

    renderer.update_scene(d, camera=cam, scene_option=vopt)
    renderer.enable_segmentation_rendering()
    seg = renderer.render()

    mask_m = np.zeros((480, 640), dtype=bool)
    for i in range(m.ngeom):
        if m.geom_group[i] == 1:
            mask_m = mask_m | ((seg[:,:,0] == i) & (seg[:,:,1] == 5))

    # Evih Mask
    mask_e_img = cv2.imread(rf'D:\AnimationTech-learning\EvihAnimation-motionbricks-replay\Demos\MotionBricksReplay\frames\frame_{frame_idx:04d}.png')
    mask_e = ((mask_e_img[:,:,0] < 250) | (mask_e_img[:,:,1] < 250) | (mask_e_img[:,:,2] < 250))

    iou = np.sum(mask_m & mask_e) / np.sum(mask_m | mask_e)
    print(f'Frame {frame_idx} IoU:', iou)
    print(f'  MuJoCo Area:', np.sum(mask_m), 'Evih Area:', np.sum(mask_e))

overlay = np.zeros((480, 640, 3), dtype=np.uint8)
overlay[mask_m] = [0, 0, 255] # MuJoCo red
overlay[mask_e] = [0, 255, 0] # Evih green
overlay[mask_m & mask_e] = [0, 255, 255] # Intersection yellow
cv2.imwrite('final_perfect_overlay_0.96.png', overlay)
cv2.imwrite(r'C:\Users\l3d\.gemini\antigravity\brain\15989d56-c4aa-4e43-b36a-5a49b647bd13\final_perfect_overlay_0.96.png', overlay)

overlay = np.zeros((480, 640, 3), dtype=np.uint8)
overlay[mask_m] = [0, 0, 255] # Red for MuJoCo
overlay[mask_e] = [0, 255, 0] # Green for Evih
overlap = (mask_m) & (mask_e)
overlay[overlap] = [0, 255, 255] # Yellow for both
cv2.imwrite('final_perfect_overlay.png', overlay)
