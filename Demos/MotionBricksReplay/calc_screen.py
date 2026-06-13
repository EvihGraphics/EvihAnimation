import numpy as np
pos_e = np.load('evih_geom_pos.npy')
cam_pos = np.array([-2.38863349,  1.6768198 ,  0.6791476])
cam_lookat = np.array([0.43044427, 0.65075941, 0.71314758])
cam_up = np.array([0,1,0])

# View Matrix
cam_z = cam_pos - cam_lookat
cam_z /= np.linalg.norm(cam_z)
cam_x = np.cross(cam_up, cam_z)
cam_x /= np.linalg.norm(cam_x)
cam_y = np.cross(cam_z, cam_x)

view_matrix = np.eye(4)
view_matrix[0, :3] = cam_x
view_matrix[1, :3] = cam_y
view_matrix[2, :3] = cam_z
view_matrix[0, 3] = -np.dot(cam_x, cam_pos)
view_matrix[1, 3] = -np.dot(cam_y, cam_pos)
view_matrix[2, 3] = -np.dot(cam_z, cam_pos)

# Projection Matrix
m0 = 2.453139305114746
m5 = 3.270852565765381
proj_matrix = np.zeros((4,4))
proj_matrix[0,0] = m0
proj_matrix[1,1] = m5
proj_matrix[2,2] = -1.0
proj_matrix[2,3] = -0.02
proj_matrix[3,2] = -1.0

min_y = 9999
min_geom = -1
for i in range(73):
    p = pos_e[50, i]
    if np.all(p == 0): continue
    p4 = np.array([p[0], p[1], p[2], 1.0])
    v = view_matrix @ p4
    c = proj_matrix @ v
    ndc = c[:3] / c[3]
    screen_x = (ndc[0] + 1.0) / 2.0 * 640
    screen_y = (1.0 - ndc[1]) / 2.0 * 480
    if screen_y < min_y:
        min_y = screen_y
        min_geom = i
    if screen_y < 50:
        print(f'Geom {i} is near top! Y={screen_y:.1f} X={screen_x:.1f}')
print('Min Y geom:', min_geom, 'at Y:', min_y)
