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
renderer.update_scene(d, camera=cam, scene_option=vopt)
renderer.enable_segmentation_rendering()
seg = renderer.render()

# MuJoCo seg ID is exactly geom_id
mask_m = np.zeros_like(seg[:,:,0], dtype=bool)
for i in range(m.ngeom):
    if m.geom_group[i] == 1:
        mask_m = mask_m | ((seg[:,:,0] == i) & (seg[:,:,1] == 5))

# Evih Mask
mask_e_img = cv2.imread(r'D:\AnimationTech-learning\EvihAnimation-motionbricks-replay\Demos\MotionBricksReplay\frames\frame_0050.png')
# Background is white (255, 255, 255). Keep everything that is NOT purely white.
# We also want to exclude the extremely light gray anti-aliasing pixels to match MuJoCo's hard edge.
mask_e = ((mask_e_img[:,:,0] < 250) | (mask_e_img[:,:,1] < 250) | (mask_e_img[:,:,2] < 250))

iou = np.sum(mask_m & mask_e) / np.sum(mask_m | mask_e)
print('FINAL IoU:', iou)
print('MuJoCo Mask Area:', np.sum(mask_m))
print('Evih Mask Area:', np.sum(mask_e))

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
