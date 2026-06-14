import os
import json
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import subprocess

def get_mujoco_joint_positions(qpos_seq, xml_path):
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    
    num_frames = qpos_seq.shape[0]
    num_bodies = m.nbody
    
    # Store position of all bodies for all frames
    # Shape: (frames, bodies, 3)
    positions = np.zeros((num_frames, num_bodies, 3))
    parents = m.body_parentid
    
    for i in range(num_frames):
        d.qpos[:] = qpos_seq[i]
        mujoco.mj_forward(m, d)
        positions[i] = d.xpos.copy()
        
    return positions, parents, None

def transform_mujoco_to_evih(positions):
    """
    MuJoCo: Z-up, X-forward, Y-left
    EvihAnimation: Y-up, Z-forward, X-right
    
    Mapping:
    X_evih = -Y_mujoco
    Y_evih = Z_mujoco
    Z_evih = X_mujoco
    """
    evih_positions = np.zeros_like(positions)
    evih_positions[:, :, 0] = -positions[:, :, 1]  # X_e = -Y_m
    evih_positions[:, :, 1] = positions[:, :, 2]   # Y_e = Z_m
    evih_positions[:, :, 2] = positions[:, :, 0]   # Z_e = X_m
    return evih_positions

def render_skeleton_mp4(evih_positions, parents, output_mp4):
    num_frames = evih_positions.shape[0]
    num_bodies = evih_positions.shape[1]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#202020')
    ax.set_facecolor('#202020')
    
    # Initialize lines for bones
    lines = [ax.plot([], [], [], '-', c='#00ffcc', lw=2)[0] for _ in range(num_bodies)]
    # Initialize scatter for joints
    scatter = ax.scatter([], [], [], c='#ff00ff', s=20)
    
    # Remove axis backgrounds
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.tick_params(colors='w')
    
    ax.set_title("EvihAnimation True-Skeleton Replay (Y-Up, Z-Forward)", color='w')

    def update(frame):
        # Update lines (bones)
        for i in range(1, num_bodies):
            parent = parents[i]
            if parent == 0: # World body
                continue
            p1 = evih_positions[frame, i]
            p2 = evih_positions[frame, parent]
            lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            lines[i].set_3d_properties([p1[2], p2[2]])
            
        # Update scatter (joints)
        xs = evih_positions[frame, 1:, 0]
        ys = evih_positions[frame, 1:, 1]
        zs = evih_positions[frame, 1:, 2]
        scatter._offsets3d = (xs, ys, zs)
        
        # Camera Tracking: Center the plot on the root (pelvis) which is index 1
        root_pos = evih_positions[frame, 1]
        window_size = 1.0
        ax.set_xlim(root_pos[0] - window_size, root_pos[0] + window_size)
        ax.set_ylim(root_pos[1] - window_size, root_pos[1] + window_size)
        ax.set_zlim(root_pos[2] - window_size, root_pos[2] + window_size)
        
        ax.set_xlabel('X (Right)')
        ax.set_ylabel('Y (Up)')
        ax.set_zlabel('Z (Forward)')
        
        return lines + [scatter]

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=33, blit=False)
    
    # Save as MP4
    writer = animation.FFMpegWriter(fps=30, bitrate=2000)
    ani.save(str(output_mp4), writer=writer)
    plt.close(fig)

def main():
    base_dir = Path(r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl")
    export_dir = base_dir / "output/motionbricks_lite/exports/demo_run_01"
    qpos_file = export_dir / "qpos.npy"
    
    if not qpos_file.exists():
        print("No qpos.npy found.")
        return
        
    qpos_seq = np.load(str(qpos_file))
    xml_path = base_dir / "motionbricks/assets/skeletons/g1/scene_29dof.xml"
    
    print("Computing MuJoCo FK...")
    mj_positions, parents, names = get_mujoco_joint_positions(qpos_seq, xml_path)
    
    print("Applying EvihAnimation Coordinate Transformation...")
    evih_positions = transform_mujoco_to_evih(mj_positions)
    
    mp4_path = export_dir / "evihanimation_true_skeleton_replay.mp4"
    print(f"Rendering tracked skeleton replay to {mp4_path}...")
    render_skeleton_mp4(evih_positions, parents, mp4_path)
    print("Done!")

if __name__ == "__main__":
    main()
