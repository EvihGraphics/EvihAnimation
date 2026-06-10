import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ai4animation import Motion, Transform, Tensor
from MotionMatching import MotionDatabase, Player, spring_character_update, simple_spring_damper_exact_quat, quat_mul_vec, quat_fk

SCRIPT_DIR = Path(__file__).parent

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
    
    # We will simulate user input to drive the character in a circle
    generated_poses = []
    parents = motions[0].Hierarchy.ParentIndices
    
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
        # Simulate gamepad input (moving in a circle)
        t = frame / 30.0
        # Slowly rotating input vector
        controller_orient = np.array([np.cos(t * 0.5), 0, np.sin(t * 0.5), 0], dtype=np.float32)
        desired_orientation = controller_orient.copy()
        magnitude = max_speed * 100
        
        desired_velocity = quat_mul_vec(controller_orient, np.array([0,0,1], dtype=np.float32)) * magnitude
        
        # Spring update
        x, v, a = spring_character_update(x, v, a, desired_velocity, halflife, 1.0/30.0)
        x_rot, v_rot = simple_spring_damper_exact_quat(x_rot, v_rot, desired_orientation, halflife_rot, 1.0/30.0)
        
        # Future trajectory for query
        f1p, _, _ = spring_character_update(x, v, a, desired_velocity, halflife, 10.0/30.0)
        f2p, _, _ = spring_character_update(x, v, a, desired_velocity, halflife, 20.0/30.0)
        f3p, _, _ = spring_character_update(x, v, a, desired_velocity, halflife, 30.0/30.0)
        
        f1q, _ = simple_spring_damper_exact_quat(x_rot, v_rot, desired_orientation, halflife_rot, 10.0/30.0)
        f2q, _ = simple_spring_damper_exact_quat(x_rot, v_rot, desired_orientation, halflife_rot, 20.0/30.0)
        f3q, _ = simple_spring_damper_exact_quat(x_rot, v_rot, desired_orientation, halflife_rot, 30.0/30.0)
        
        # Transform future to local space of current spring character
        from MotionMatching import quat_inv
        inv_q = quat_inv(x_rot)
        inv_p = -x
        
        f1p_loc = quat_mul_vec(inv_q, f1p + inv_p)
        f2p_loc = quat_mul_vec(inv_q, f2p + inv_p)
        f3p_loc = quat_mul_vec(inv_q, f3p + inv_p)
        
        # Build query
        query_vector = db.features_normalized[player.frame, :].copy()
        
        # We need to un-normalize it first to inject new values, then re-normalize.
        # But wait! It's easier to build raw feature and then normalize:
        raw_query = db.features[player.frame, :].copy()
        raw_query[15:18] = f1p_loc
        raw_query[18:21] = f2p_loc
        raw_query[21:24] = f3p_loc
        
        # Future directions
        raw_query[24:27] = quat_mul_vec(inv_q, quat_mul_vec(f1q, np.array([0,0,1], dtype=np.float32)))
        raw_query[27:30] = quat_mul_vec(inv_q, quat_mul_vec(f2q, np.array([0,0,1], dtype=np.float32)))
        raw_query[30:33] = quat_mul_vec(inv_q, quat_mul_vec(f3q, np.array([0,0,1], dtype=np.float32)))
        
        query_normalized = (raw_query - db.features_mean) / db.features_std
        
        # Search
        best_frame = db.query(np.expand_dims(query_normalized, axis=0))[0]
        
        # We don't jump every frame, usually we jump if cost is high or after N frames.
        # Let's force jump if best_frame is far from current frame
        if abs(best_frame - player.frame) > 10:
            player.set_next_frame(best_frame + 1, inertialize=True)
        else:
            player.set_next_frame(player.frame + 1, inertialize=True)
            
        # Align simulation x to animation p[0] to prevent drift
        x = 0.9 * x + 0.1 * player.p[0]
        player.p[0] = x
        
        # Generate world positions for rendering
        q = player.q
        p = player.p
        
        
        q_evih = np.zeros_like(q)
        q_evih[..., 0:3] = q[..., 1:4]
        q_evih[..., 3] = q[..., 0]
        
        # Calculate global positions via Forward Kinematics
        g_q, g_p = quat_fk(np.expand_dims(q, axis=0), np.expand_dims(p, axis=0), parents)
        
        # Just use global positions directly for the plot
        positions = g_p[0]
        generated_poses.append(positions)
        print(f"Rendered frame {frame}/{num_frames}")

    print("Saving GIF...")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    bones = []
    for i in range(1, motions[0].NumJoints):
        if parents[i] >= 0:
            bones.append((i, parents[i]))
            
    def update(frame):
        ax.clear()
        pos = generated_poses[frame]
        
        root = pos[0]
        ax.set_xlim3d([root[0] - 2, root[0] + 2])
        ax.set_ylim3d([root[2] - 2, root[2] + 2])
        ax.set_zlim3d([0, 2])
        ax.view_init(elev=20., azim=-45 + frame*0.1)
        
        xs = pos[:, 0]
        ys = pos[:, 2]
        zs = pos[:, 1]
        
        for c, p_idx in bones:
            ax.plot([xs[c], xs[p_idx]], [ys[c], ys[p_idx]], [zs[c], zs[p_idx]], 'b-', lw=2)
            
        ax.scatter(xs, ys, zs, c='r', s=10)
        ax.set_title(f"Motion Matching - Frame {frame}")
        return ax,
        
    anim = animation.FuncAnimation(fig, update, frames=len(generated_poses), interval=33, blit=False)
    out_path = str(SCRIPT_DIR / "motion_matching_demo.gif")
    anim.save(out_path, fps=30)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
