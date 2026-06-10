import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from ai4animation import Motion
from ai4animation.Math import Transform as T
from MotionGraph import compute_distance_matrix, extract_local_minima, MotionGraph

SCRIPT_DIR = Path(__file__).parent
BVH_PATH = str(SCRIPT_DIR.parent.parent.parent / "resources" / "lafan1" / "bvh" / "walk1_subject5.bvh")

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
    
    print("Generating motion sequence...")
    # Generate 300 frames of motion traversing the graph
    current_node = list(mg.scc_nodes)[0]
    
    # Track accumulated global transform to apply continuous movement
    accumulated_transform = np.eye(4)
    
    generated_poses = []
    
    for i in range(300):
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
        
        # Get local pose
        local_pose_transforms = motion.Frames[current_node]
        
        # Apply accumulated root transform to the whole pose
        global_pose = np.zeros_like(local_pose_transforms)
        for j in range(local_pose_transforms.shape[0]):
            global_pose[j] = accumulated_transform @ local_pose_transforms[j]
            
        # Extract positions
        positions = np.zeros((global_pose.shape[0], 3))
        for j in range(global_pose.shape[0]):
            positions[j] = global_pose[j][:3, 3]
            
        generated_poses.append(positions)

    print("Rendering video...")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Definition of skeleton connections for standard CMU/LAFAN1
    # Simple heuristic to draw lines between parent-child
    bones = []
    parents = motion.Hierarchy.ParentIndices
    for i in range(1, motion.NumJoints):
        parent = parents[i]
        if parent >= 0:
            bones.append((i, parent))
            
    lines = [ax.plot([], [], [], '-', c='blue', lw=2)[0] for _ in bones]
    points, = ax.plot([], [], [], 'o', c='red', markersize=3)
    
    def update(frame):
        ax.clear()
        ax.set_xlim3d([-2 + frame*0.01, 2 + frame*0.01])
        ax.set_ylim3d([-2, 2])
        ax.set_zlim3d([0, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y (Up)')
        ax.view_init(elev=20., azim=-45 + frame*0.1)
        
        # Note: BVH Y is up, X is right, Z is forward
        # matplotlib 3D puts Z as up, so we map X->X, Y->Z, Z->Y
        pos = generated_poses[frame]
        
        xs = pos[:, 0]
        ys = pos[:, 2]  # depth
        zs = pos[:, 1]  # up
        
        for idx, (c, p) in enumerate(bones):
            ax.plot([xs[c], xs[p]], [ys[c], ys[p]], [zs[c], zs[p]], 'b-', lw=2)
            
        ax.scatter(xs, ys, zs, c='r', s=10)
        ax.set_title(f"Motion Graph Traversal - Frame {frame}")
        return ax,
        
    anim = animation.FuncAnimation(fig, update, frames=len(generated_poses), interval=33, blit=False)
    
    out_path = str(SCRIPT_DIR / "motion_graph_demo.gif")
    # Save as gif using pillow
    anim.save(out_path, fps=30)
    print(f"Video saved to {out_path}")

if __name__ == "__main__":
    main()
