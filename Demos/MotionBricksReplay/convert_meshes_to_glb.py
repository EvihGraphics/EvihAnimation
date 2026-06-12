import os
import trimesh
import json
import numpy as np

def convert_meshes():
    dest_dir = "meshes"
        
    with open("geom_mesh_names.json", "r") as f:
        mesh_names = json.load(f)
        
    unique_meshes = set(name for name in mesh_names if name and "logo" not in name)
    print(f"Found {len(unique_meshes)} unique meshes to convert")
    
    for mesh_name in unique_meshes:
        obj_path = os.path.join(dest_dir, f"{mesh_name}.obj")
        glb_path = os.path.join(dest_dir, f"{mesh_name}.glb")
        
        if not os.path.exists(obj_path):
            print(f"Warning: Source OBJ not found: {obj_path}")
            continue
            
        print(f"Converting {mesh_name}...")
        try:
            mesh = trimesh.load(obj_path)
            mesh.export(glb_path, file_type='glb')
            print(f"  Successfully converted to {glb_path}")
        except Exception as e:
            print(f"  Failed to convert {mesh_name}: {e}")

if __name__ == "__main__":
    convert_meshes()
