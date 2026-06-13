import mujoco
import numpy as np
import trimesh
import os

m = mujoco.MjModel.from_xml_path(r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\scene_29dof.xml')

output_dir = 'meshes_mujoco'
os.makedirs(output_dir, exist_ok=True)

T_m_to_e = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

R_m_to_e = T_m_to_e[:3, :3]

# In scene_29dof.xml, visual meshes are in group 1
for i in range(m.ngeom):
    if m.geom_group[i] != 1:
        continue
    if m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
        
    mesh_id = m.geom_dataid[i]
    name = f'Geom_{i}'
    
    vert_adr = m.mesh_vertadr[mesh_id]
    num_vert = m.mesh_vertnum[mesh_id]
    verts = m.mesh_vert[vert_adr:vert_adr+num_vert]
    
    face_adr = m.mesh_faceadr[mesh_id]
    num_face = m.mesh_facenum[mesh_id]
    faces = m.mesh_face[face_adr:face_adr+num_face]
    
    # Transform to Evih coordinate system
    verts_e = np.dot(verts, R_m_to_e.T)
    
    # Invert faces to maintain correct winding order after reflection/axis swap
    # T_m_to_e swaps X->Z, Y->X, Z->Y. The determinant is:
    # | 0 1 0 |
    # | 0 0 1 | = 0*(0-0) - 1*(0-1) + 0 = 1
    # Determinant is 1, so it is a PURE ROTATION.
    # Therefore, winding order DOES NOT need to be inverted!
    
    mesh = trimesh.Trimesh(vertices=verts_e, faces=faces)
    mesh.export(os.path.join(output_dir, f'{name}.glb'), file_type='glb')
    print(f'Exported {name}.glb')
