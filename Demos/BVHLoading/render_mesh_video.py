import sys, os
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from ai4animation import Motion, Transform
from ai4animation.Import.GLBImporter import GLB
from ai4animation.Standalone.OfflineMeshRenderer import OfflineMeshRenderer

MOTION_FILE = "WalkingStickLeft_BR"

def extract_local_matrices(global_matrices, parent_indices):
    local_matrices = np.zeros_like(global_matrices)
    for i in range(len(global_matrices)):
        p = parent_indices[i]
        if p == -1:
            local_matrices[i] = global_matrices[i]
        else:
            local_matrices[i] = np.linalg.inv(global_matrices[p]) @ global_matrices[i]
    return local_matrices

def main():
    bvh_path = os.path.join(str(SCRIPT_DIR), MOTION_FILE + ".bvh")
    print(f"Loading BVH: {bvh_path}")
    motion = Motion.LoadFromBVH(bvh_path, scale=0.01)
    
    print("Loading GLB Model...")
    glb_path = str(SCRIPT_DIR.parent / "_ASSETS_" / "Geno" / "Model.glb")
    glb = GLB.Create(glb_path)
    
    # Pre-map BVH bones to GLB nodes
    bvh_names = motion.Hierarchy.BoneNames
    glb_names = glb._nodeNames
    bvh_to_glb = {}
    for i, name in enumerate(bvh_names):
        if name in glb_names:
            bvh_to_glb[i] = glb_names.index(name)
            
    # Base GLB local matrices from the zero pose
    base_glb_local = np.zeros((len(glb_names), 4, 4), dtype=np.float32)
    for i, node in enumerate(glb._nodes):
        base_glb_local[i] = node.LocalMatrix
        
    global_matrices_seq = []
    
    num_frames = min(300, motion.NumFrames)
    
    for frame in range(num_frames):
        bvh_global_pose = motion.Frames[frame]
        bvh_local_pose = extract_local_matrices(bvh_global_pose, motion.Hierarchy.ParentIndices)
        
        # Map to GLB
        glb_local = base_glb_local.copy()
        for bvh_idx, glb_idx in bvh_to_glb.items():
            glb_local[glb_idx] = bvh_local_pose[bvh_idx]
            
        # FK on GLB hierarchy
        glb_global = np.zeros_like(glb_local)
        for i, node in enumerate(glb._nodes):
            if node.Parent is None:
                glb_global[i] = glb_local[i]
            else:
                glb_global[i] = glb_global[node.Parent] @ glb_local[i]
                
        global_matrices_seq.append(glb_global)

    # Render Sequence
    renderer = OfflineMeshRenderer(glb, width=1280, height=720)
    out_path = str(SCRIPT_DIR / "bvhloading_mesh.mp4")
    renderer.render_animation(global_matrices_seq, out_path, fps=30)

if __name__ == "__main__":
    main()
