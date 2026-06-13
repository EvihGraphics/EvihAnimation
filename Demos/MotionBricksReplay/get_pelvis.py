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
m.vis.global_.fovy = 34.0
renderer = mujoco.Renderer(m, 480, 640)
for i in range(m.ngeom):
    if m.geom_group[i] != 1:
        m.geom_pos[i] = [0, 0, 1000]
mujoco.mj_forward(m, d)
renderer.update_scene(d, camera=cam)
renderer.enable_segmentation_rendering()
seg = renderer.render()
img = cv2.imread(r'frames\frame_0051.png')
mask_e = ((img[:,:,0]<50) & (img[:,:,1]<50) & (img[:,:,2]<50))
y,x = np.where(mask_e)
if len(y) > 0:
    for ty in [0, 1, 2]:
        print(f'At Y={ty}: X=', x[y==ty])
