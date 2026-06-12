import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def render_skeleton_mp4(positions_file, parents_file, output_mp4):
    evih_positions = np.load(positions_file)
    parents = np.load(parents_file)
    
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
    
    ax.set_title("EvihAnimation Skeleton Replay (Y-Up, Z-Forward)", color='w')

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

    print(f"Generating MP4 offline rendering ({num_frames} frames)...")
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=33, blit=False)
    
    writer = animation.FFMpegWriter(fps=30, bitrate=2000)
    ani.save(str(output_mp4), writer=writer)
    plt.close(fig)
    print(f"Saved {output_mp4}")

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "evih_skeleton_offline.mp4")
    render_skeleton_mp4("evih_positions.npy", "parents.npy", output_path)
