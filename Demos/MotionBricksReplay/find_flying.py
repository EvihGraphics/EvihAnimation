import numpy as np
import trimesh
import json
import os

pos_e = np.load('evih_geom_pos.npy')
rot_e = np.load('evih_geom_rot.npy')
with open('geom_mesh_names.json', 'r') as f:
    geom_mesh_names = json.load(f)

cam_pos = np.array([-2.38863349,  1.6768198 ,  0.6791476])
cam_lookat = np.array([0.43044427, 0.65075941, 0.71314758])
cam_up = np.array([0,1,0])

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

m0 = 2.453139305114746
m5 = 3.270852565765381
proj_matrix = np.zeros((4,4))
proj_matrix[0,0] = m0
proj_matrix[1,1] = m5
proj_matrix[2,2] = -1.0
proj_matrix[2,3] = -0.02
proj_matrix[3,2] = -1.0

for i, name in enumerate(geom_mesh_names):
    if not name: continue
    obj_path = f"meshes/{name}.obj"
    if not os.path.exists(obj_path): continue
    
    mesh = trimesh.load(obj_path)
    v = mesh.vertices
    
    p = pos_e[50, i]
    R = rot_e[50, i]
    
    v_world = p + np.dot(v, R.T)
    v4 = np.hstack((v_world, np.ones((len(v), 1))))
    
    c = (proj_matrix @ view_matrix @ v4.T).T
    ndc = c[:, :3] / c[:, 3:4]
    
    screen_x = (ndc[:, 0] + 1.0) / 2.0 * 640
    screen_y = (1.0 - ndc[:, 1]) / 2.0 * 480
    
    min_y = np.min(screen_y)
    max_y = np.max(screen_y)
    if min_y < 50 or max_y > 450:
        print(f'Geom {i} ({name}) Min Y: {min_y:.1f}, Max Y: {max_y:.1f}')
