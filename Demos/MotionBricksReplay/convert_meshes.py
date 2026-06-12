import os
import glob
import trimesh

def convert_stl_to_obj(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    stl_files = glob.glob(os.path.join(input_dir, "*.STL"))
    print(f"Found {len(stl_files)} STL files to convert.")
    for stl_path in stl_files:
        basename = os.path.basename(stl_path)
        obj_name = basename.replace(".STL", ".obj")
        obj_path = os.path.join(output_dir, obj_name)
        
        # Load mesh
        mesh = trimesh.load_mesh(stl_path)
        
        # In MuJoCo, these STLs are in meters but trimesh might not scale them.
        # STLs do not have units, but MuJoCo xml scales them or uses them as-is.
        # We will export to OBJ as-is.
        mesh.export(obj_path)
        print(f"Converted {basename} to {obj_name}")

if __name__ == "__main__":
    input_dir = r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\meshes"
    output_dir = r"D:\AnimationTech-learning\EvihAnimation-motionbricks-replay\Demos\MotionBricksReplay\meshes"
    convert_stl_to_obj(input_dir, output_dir)
