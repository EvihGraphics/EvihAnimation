import os
import glob
import trimesh
import numpy as np

def convert_stl_to_obj(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    stl_files = glob.glob(os.path.join(input_dir, "*.STL"))
    print(f"Found {len(stl_files)} STL files to convert.")
    
    # Basis Transformation Matrix: MuJoCo (Z-Up) to Evih (Y-Up)
    T_m_to_e = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    for stl_path in stl_files:
        basename = os.path.basename(stl_path)
        glb_name = basename.replace(".STL", ".glb")
        glb_path = os.path.join(output_dir, glb_name)
        
        mesh = trimesh.load_mesh(stl_path)
        
        if hasattr(mesh, 'apply_transform'):
            mesh.apply_transform(T_m_to_e)
        elif hasattr(mesh, 'geometry'): 
            for geom in mesh.geometry.values():
                geom.apply_transform(T_m_to_e)
                
        if hasattr(mesh, 'fix_normals'):
            mesh.fix_normals()
        else:
            for geom in mesh.geometry.values():
                geom.fix_normals()
                
        mesh.export(glb_path, file_type='glb')
        print(f"Converted {basename} to {glb_name}")

if __name__ == "__main__":
    input_dir = r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\meshes"
    output_dir = r"D:\AnimationTech-learning\EvihAnimation-motionbricks-replay\Demos\MotionBricksReplay\meshes"
    convert_stl_to_obj(input_dir, output_dir)
