import numpy as np
import trimesh
import json
import os

def main():
    geom_pos = np.load('evih_geom_pos.npy')
    geom_rot = np.load('evih_geom_rot.npy')
    
    with open('geom_mesh_names.json', 'r') as f:
        geom_mesh_names = json.load(f)
        
    frame = 0
    all_vertices = []
    
    for i, name in enumerate(geom_mesh_names):
        if not name: continue
        obj_path = f"meshes/{name}.obj"
        if not os.path.exists(obj_path): continue
        
        mesh = trimesh.load(obj_path)
        # vertices are already transformed by T_m_to_e in convert_meshes.py
        # meaning they are in Evih local space
        v = mesh.vertices
        
        p = geom_pos[frame, i]
        R = geom_rot[frame, i]
        
        # Apply Evih transform: v_world = p + R * v
        v_world = p + np.dot(v, R.T)
        
        all_vertices.append(v_world)
        
    all_vertices = np.vstack(all_vertices)
    
    # Save the combined mesh as a point cloud or just a single OBJ
    combined = trimesh.Trimesh(vertices=all_vertices, faces=np.zeros((0,3)))
    # Wait, exporting point cloud might not work nicely, let's just save combined.obj with all faces!
    
    combined_mesh = trimesh.Scene()
    for i, name in enumerate(geom_mesh_names):
        if not name: continue
        obj_path = f"meshes/{name}.obj"
        if not os.path.exists(obj_path): continue
        
        mesh = trimesh.load(obj_path)
        p = geom_pos[frame, i]
        R = geom_rot[frame, i]
        
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = p
        
        mesh.apply_transform(transform)
        combined_mesh.add_geometry(mesh)
        
    combined_mesh.export('evih_frame0_combined.obj')
    print("Exported evih_frame0_combined.obj")

if __name__ == "__main__":
    main()
