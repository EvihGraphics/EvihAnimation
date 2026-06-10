import sys, os
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from ai4animation import Motion, Tensor
from ai4animation.Import.GLBImporter import GLB
from MotionMatching import MotionDatabase, Player, spring_character_update, simple_spring_damper_exact_quat, quat_mul_vec
from ai4animation.Standalone.OfflineMeshRenderer import OfflineMeshRenderer

def build_matrix(q, p):
    # q is [w, x, y, z]
    qw, qx, qy, qz = q
    mat = np.eye(4, dtype=np.float32)
    mat[0, 0] = 1 - 2*qy*qy - 2*qz*qz
    mat[0, 1] = 2*qx*qy - 2*qz*qw
    mat[0, 2] = 2*qx*qz + 2*qy*qw
    
    mat[1, 0] = 2*qx*qy + 2*qz*qw
    mat[1, 1] = 1 - 2*qx*qx - 2*qz*qz
    mat[1, 2] = 2*qy*qz - 2*qx*qw
    
    mat[2, 0] = 2*qx*qz - 2*qy*qw
    mat[2, 1] = 2*qy*qz + 2*qx*qw
    mat[2, 2] = 1 - 2*qx*qx - 2*qy*qy
    
    mat[:3, 3] = p
    return mat

def main():
    bvh_dir = SCRIPT_DIR.parent.parent.parent / "resources" / "lafan1" / "bvh"
    bvh_paths = [
        str(bvh_dir / "run2_subject4.bvh"),
        str(bvh_dir / "run1_subject2.bvh"),
        str(bvh_dir / "walk1_subject5.bvh")
    ]
    
    motions = []
    for path in bvh_paths:
        print(f"Loading {os.path.basename(path)}...")
        m = Motion.LoadFromBVH(path, scale=0.01)
        if m.NumFrames > 1000: m.Frames = m.Frames[:1000]
        motions.append(m)
        
    db = MotionDatabase()
    db.build_from_motions(motions)
    player = Player(db)
    
    # Load GLB Model
    glb_path = str(SCRIPT_DIR.parent / "_ASSETS_" / "Geno" / "Model.glb")
    print(f"Loading GLB Model from {glb_path}...")
    glb = GLB.Create(glb_path)
    
    # Pre-map BVH bones to GLB nodes
    bvh_names = motions[0].Hierarchy.BoneNames
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
    
    # Simulation state
    x = np.zeros(3, dtype=np.float32)
    v = np.zeros(3, dtype=np.float32)
    a = np.zeros(3, dtype=np.float32)
    
    x_rot = np.array([1,0,0,0], dtype=np.float32)
    v_rot = np.zeros(3, dtype=np.float32)
    desired_orientation = np.array([1,0,0,0], dtype=np.float32)
    
    max_speed = 3.5
    halflife = 0.3
    halflife_rot = 0.3
    
    num_frames = 300
    for frame in range(num_frames):
        t = frame / 30.0
        controller_orient = np.array([np.cos(t * 0.5), 0, np.sin(t * 0.5), 0], dtype=np.float32)
        desired_orientation = controller_orient.copy()
        magnitude = max_speed * 100
        desired_velocity = quat_mul_vec(controller_orient, np.array([0,0,1], dtype=np.float32)) * magnitude
        
        # Spring update
        x, v, a = spring_character_update(x, v, a, desired_velocity, halflife, 1.0/30.0)
        x_rot, v_rot = simple_spring_damper_exact_quat(x_rot, v_rot, desired_orientation, halflife_rot, 1.0/30.0)
        
        # Future predictions
        f1p, _, _ = spring_character_update(x, v, a, desired_velocity, halflife, 10.0/30.0)
        f2p, _, _ = spring_character_update(x, v, a, desired_velocity, halflife, 20.0/30.0)
        f3p, _, _ = spring_character_update(x, v, a, desired_velocity, halflife, 30.0/30.0)
        
        f1q, _ = simple_spring_damper_exact_quat(x_rot, v_rot, desired_orientation, halflife_rot, 10.0/30.0)
        f2q, _ = simple_spring_damper_exact_quat(x_rot, v_rot, desired_orientation, halflife_rot, 20.0/30.0)
        f3q, _ = simple_spring_damper_exact_quat(x_rot, v_rot, desired_orientation, halflife_rot, 30.0/30.0)
        
        from MotionMatching import quat_inv
        inv_q = quat_inv(x_rot)
        inv_p = -x
        
        f1p_loc = quat_mul_vec(inv_q, f1p + inv_p)
        f2p_loc = quat_mul_vec(inv_q, f2p + inv_p)
        f3p_loc = quat_mul_vec(inv_q, f3p + inv_p)
        
        # Build query
        raw_query = db.features[player.frame, :].copy()
        raw_query[15:18] = f1p_loc
        raw_query[18:21] = f2p_loc
        raw_query[21:24] = f3p_loc
        
        raw_query[24:27] = quat_mul_vec(inv_q, quat_mul_vec(f1q, np.array([0,0,1], dtype=np.float32)))
        raw_query[27:30] = quat_mul_vec(inv_q, quat_mul_vec(f2q, np.array([0,0,1], dtype=np.float32)))
        raw_query[30:33] = quat_mul_vec(inv_q, quat_mul_vec(f3q, np.array([0,0,1], dtype=np.float32)))
        
        query_normalized = (raw_query - db.features_mean) / db.features_std
        best_frame = db.query(np.expand_dims(query_normalized, axis=0))[0]
        
        if abs(best_frame - player.frame) > 10:
            player.set_next_frame(best_frame + 1, inertialize=True)
        else:
            player.set_next_frame(player.frame + 1, inertialize=True)
            
        x = 0.9 * x + 0.1 * player.p[0]
        player.p[0] = x
        
        # Now apply player.q and player.p to GLB local matrices
        glb_local = base_glb_local.copy()
        for bvh_idx, glb_idx in bvh_to_glb.items():
            glb_local[glb_idx] = build_matrix(player.q[bvh_idx], player.p[bvh_idx])
            
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
    out_path = str(SCRIPT_DIR / "motion_matching_mesh.mp4")
    renderer.render_animation(global_matrices_seq, out_path, fps=30)

if __name__ == "__main__":
    main()
