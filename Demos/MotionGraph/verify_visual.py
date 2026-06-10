"""
Motion Graph Visual Verification Script
Generates distance matrix heatmap + local minima overlay for comparison with original notebook.
Runs headless (no GUI needed).
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

from ai4animation import Motion
from MotionGraph import compute_distance_matrix, extract_local_minima, MotionGraph

SCRIPT_DIR = Path(__file__).parent
BVH_PATH = str(SCRIPT_DIR.parent.parent.parent / "resources" / "lafan1" / "bvh" / "walk1_subject5.bvh")
OUT_DIR = SCRIPT_DIR

def main():
    print("Loading BVH...")
    motion = Motion.LoadFromBVH(BVH_PATH, scale=0.01)
    
    MAX_FRAMES = 600
    if motion.NumFrames > MAX_FRAMES:
        print(f"Slicing motion from {motion.NumFrames} to {MAX_FRAMES} frames")
        motion.Frames = motion.Frames[:MAX_FRAMES]
    
    motion.Debug()
    
    # --- Compute Distance Matrix ---
    print("\nComputing distance matrix...")
    dist_mat, transform_mat = compute_distance_matrix(motion, window_size=15)
    
    # --- Plot 1: Distance Matrix Heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Raw distance matrix
    im0 = axes[0].imshow(np.clip(dist_mat, 0, 0.5), cmap='hot', aspect='auto', origin='lower')
    axes[0].set_title('Distance Matrix (clipped to [0, 0.5])', fontsize=12)
    axes[0].set_xlabel('Frame j')
    axes[0].set_ylabel('Frame i')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # --- Extract Local Minima ---
    minima = extract_local_minima(dist_mat, threshold=0.1, local_window=10)
    
    # Plot 2: Distance matrix with local minima overlay
    im1 = axes[1].imshow(np.clip(dist_mat, 0, 0.5), cmap='gray', aspect='auto', origin='lower')
    if len(minima) > 0:
        mi = [m[0] for m in minima]
        mj = [m[1] for m in minima]
        axes[1].scatter(mj, mi, c='red', s=5, alpha=0.8, label=f'{len(minima)} minima')
        axes[1].legend(loc='upper right', fontsize=10)
    axes[1].set_title('Local Minima (potential transitions)', fontsize=12)
    axes[1].set_xlabel('Frame j')
    axes[1].set_ylabel('Frame i')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # --- Build Graph and show SCC ---
    mg = MotionGraph(motion)
    mg.build(dist_mat, transform_mat, minima)
    
    # Plot 3: SCC membership
    scc_mask = np.zeros(motion.NumFrames)
    for n in mg.scc_nodes:
        scc_mask[n] = 1
    
    # Show which frames are in SCC
    axes[2].bar(range(motion.NumFrames), scc_mask, width=1.0, color='green', alpha=0.6)
    axes[2].set_title(f'SCC Membership ({len(mg.scc_nodes)}/{motion.NumFrames} frames)', fontsize=12)
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('In SCC')
    axes[2].set_ylim(-0.1, 1.1)
    
    # Count edges per frame
    edge_counts = np.zeros(motion.NumFrames)
    for frame_id, edges in mg.edges.items():
        jump_edges = [e for e in edges if e['type'] == 'jump']
        edge_counts[frame_id] = len(jump_edges)
    
    ax2_twin = axes[2].twinx()
    ax2_twin.bar(range(motion.NumFrames), edge_counts, width=1.0, color='red', alpha=0.3)
    ax2_twin.set_ylabel('Jump Edges', color='red')
    
    plt.tight_layout()
    out_path = str(OUT_DIR / "visual_verification.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved verification plot to: {out_path}")
    
    # --- Graph traversal statistics ---
    print("\n=== Motion Graph Statistics ===")
    print(f"Total frames: {motion.NumFrames}")
    print(f"Local minima found: {len(minima)}")
    print(f"SCC size: {len(mg.scc_nodes)} / {motion.NumFrames}")
    
    total_jump_edges = sum(1 for edges in mg.edges.values() for e in edges if e['type'] == 'jump')
    total_seq_edges = sum(1 for edges in mg.edges.values() for e in edges if e['type'] == 'seq')
    print(f"Sequential edges: {total_seq_edges}")
    print(f"Jump edges: {total_jump_edges}")
    
    # Simulate 500 steps
    print("\n=== Simulating 500 graph traversal steps ===")
    current = list(mg.scc_nodes)[0]
    jump_count = 0
    seq_count = 0
    for _ in range(500):
        next_frame, trans_type, _ = mg.get_next_state(current)
        if trans_type == 'jump':
            jump_count += 1
        else:
            seq_count += 1
        current = next_frame
    print(f"Sequential steps: {seq_count} ({seq_count/5:.1f}%)")
    print(f"Jump steps: {jump_count} ({jump_count/5:.1f}%)")
    print(f"Average clip length before jump: {seq_count/max(1,jump_count):.1f} frames")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
