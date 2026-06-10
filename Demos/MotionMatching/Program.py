import os
import sys
from pathlib import Path
import numpy as np

# Add the framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai4animation import (
    AI4Animation,
    Actor,
    Vector3,
    Time,
    Motion,
    Transform,
    Tensor
)

from MotionMatching import MotionDatabase, Player

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_" / "Geno")
sys.path.append(ASSETS_PATH)
import Definitions

class Program:
    def __init__(self):
        # We will load a few run/walk BVHs to build our database
        bvh_dir = SCRIPT_DIR.parent.parent.parent / "resources" / "lafan1" / "bvh"
        self.bvh_paths = [
            str(bvh_dir / "run2_subject4.bvh"),
            str(bvh_dir / "run1_subject2.bvh"),
            str(bvh_dir / "walk1_subject5.bvh")
        ]
        self.model_path = os.path.join(ASSETS_PATH, "Model.glb")
        
        self.db = None
        self.player = None

    def Start(self):
        motions = []
        for path in self.bvh_paths:
            print(f"Loading {os.path.basename(path)}...")
            m = Motion.LoadFromBVH(path, scale=0.01)
            # Limit frames to speed up KD-tree build for testing
            if m.NumFrames > 1000:
                m.Frames = m.Frames[:1000]
            motions.append(m)
            
        print("Building Motion Matching Database...")
        self.db = MotionDatabase()
        self.db.build_from_motions(motions)
        
        self.player = Player(self.db)
        
        entity = AI4Animation.Scene.AddEntity("Character")
        self.actor = entity.AddComponent(
            Actor, self.model_path, Definitions.FULL_BODY_NAMES
        )
        
    def Standalone(self):
        AI4Animation.Standalone.Camera.SetTarget(self.actor.Entity)
        AI4Animation.Standalone.Camera.Offset = Vector3.Create(0, 1.5, 3)

    def _get_align_matrix(self, theta, tx, tz):
        mat = Transform.Identity()
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        mat[0, 0] = cos_t
        mat[0, 2] = sin_t
        mat[2, 0] = -sin_t
        mat[2, 2] = cos_t
        mat[0, 3] = tx
        mat[2, 3] = tz
        return mat

    def Update(self):
        if self.player is None:
            return
            
        # Simulate gamepad input using Keyboard in EvihAnimation
        # W/S for forward/backward, A/D for rotation or strafing
        # In Raylib Standalone, we might not have direct gamepad axes exposed via python easily,
        # so we will simulate it with a fixed trajectory for demonstration, or use basic keyboard keys if supported.
        # Let's just simulate a simple circular trajectory for demonstration
        
        t = Time.Time
        desired_velocity = np.array([np.sin(t * 0.5) * 200, 0, np.cos(t * 0.5) * 200], dtype=np.float32)
        
        # We need the spring update to get future trajectory
        from MotionMatching import spring_character_update, simple_spring_damper_exact_quat, quat_mul_vec, quat_inv
        
        # Build query vector
        query_vector = self.db.features_normalized[self.player.frame, :].copy()
        
        # Simplified query: just search based on current state (we can add future traj prediction here later)
        # Search DB
        best_frame = self.db.query(np.expand_dims(query_vector, axis=0))[0]
        
        # Step Player
        # If best frame is drastically different, it jumps. Otherwise just plays sequence.
        # But we only want to jump occasionally or if the cost is very high. 
        # For a basic test, let's just jump to the best frame every 15 frames, or if cost is low enough.
        
        if Time.Frame % 15 == 0:
            self.player.set_next_frame(best_frame + 1, inertialize=True)
        else:
            self.player.set_next_frame(self.player.frame + 1, inertialize=True)
            
        # Update character Transforms
        q = self.player.q
        p = self.player.p
        
        # Convert quat/pos to EvihAnimation 4x4 matrix
        # Note: q is [w, x, y, z] from our MotionMatching DB!
        from ai4animation.Math.Quaternion import ToMatrix
        # ai4animation expects [x,y,z,w]? Let's check: Yes, ToMatrix takes [..., 3] as w.
        # We need to swap [w,x,y,z] to [x,y,z,w] for EvihAnimation
        q_evih = np.zeros_like(q)
        q_evih[..., 0:3] = q[..., 1:4]
        q_evih[..., 3] = q[..., 0]
        
        rot_mat = ToMatrix(Tensor.Create(q_evih)) # [J, 3, 3]
        
        world_poses = np.zeros((self.actor.NumBones, 4, 4), dtype=np.float32)
        world_poses[:, 0:3, 0:3] = rot_mat
        world_poses[:, 0:3, 3] = p
        world_poses[:, 3, 3] = 1.0
        
        self.actor.SetTransforms(Tensor.Create(world_poses))
        self.actor.SyncToScene()

    def GUI(self):
        AI4Animation.Draw.Text(f"Frame: {self.player.frame if self.player else 0}", 0.02, 0.02, 0.02, AI4Animation.Color.BLACK)
        AI4Animation.Draw.Text("Motion Matching Demo", 0.02, 0.05, 0.02, AI4Animation.Color.BLUE)

if __name__ == "__main__":
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)
