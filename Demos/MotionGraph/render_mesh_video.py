import sys, os
from pathlib import Path
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from ai4animation import Motion
from ai4animation.Import.GLBImporter import GLB
from MotionGraph import compute_distance_matrix, extract_local_minima, MotionGraph
from ai4animation.Standalone.OfflineMeshRenderer import OfflineMeshRenderer

BVH_PATH = str(SCRIPT_DIR.parent.parent.parent / "resources" / "lafan1" / "bvh" / "walk1_subject5.bvh")

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
    print("Loading BVH...")
    motion = Motion.LoadFromBVH(BVH_PATH, scale=0.01)
    
    MAX_FRAMES = 600
    if motion.NumFrames > MAX_FRAMES:
        motion.Frames = motion.Frames[:MAX_FRAMES]
        
    print("Computing distance matrix...")
    dist_mat, transform_mat = compute_distance_matrix(motion, window_size=15)
    
    minima = extract_local_minima(dist_mat, threshold=0.1, local_window=10)
    mg = MotionGraph(motion)
    mg.build(dist_mat, transform_mat, minima)
    
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
        
    print("Generating motion sequence...")
    # Generate 300 frames of motion traversing the graph
    current_node = list(mg.scc_nodes)[0]
    
    # Track accumulated global transform to apply continuous movement
    accumulated_transform = np.eye(4)
    
    global_matrices_seq = []
    
    for _ in range(300):
        next_node, trans_type, local_align_mat = mg.get_next_state(current_node)
        
        if trans_type == 'jump':
            # local_align_mat is an array of shape (3,) containing (theta, tx, tz)
            theta, tx, tz = local_align_mat
            
            align_mat = np.eye(4)
            align_mat[0, 0] = np.cos(theta)
            align_mat[0, 2] = np.sin(theta)
            align_mat[2, 0] = -np.sin(theta)
            align_mat[2, 2] = np.cos(theta)
            align_mat[0, 3] = tx
            align_mat[2, 3] = tz
                
            accumulated_transform = accumulated_transform @ align_mat
            
        current_node = next_node
        
        # Get BVH local poses
        bvh_global_pose = motion.Frames[current_node]
        bvh_local_pose = extract_local_matrices(bvh_global_pose, motion.Hierarchy.ParentIndices)
        
        # Apply accumulated root transform to the root local pose
        # Since root's local pose is its global pose, we just multiply by accumulated_transform
        root_idx = 0
        bvh_local_pose[root_idx] = accumulated_transform @ bvh_local_pose[root_idx]
        
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
    out_path = str(SCRIPT_DIR / "motion_graph_mesh.mp4")
    renderer.render_animation(global_matrices_seq, out_path, fps=30)

if __name__ == "__main__":
    main()
