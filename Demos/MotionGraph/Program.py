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

from MotionGraph import compute_distance_matrix, extract_local_minima, MotionGraph

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_" / "Geno")
sys.path.append(ASSETS_PATH)
import Definitions

class Program:
    def __init__(self):
        # We will use the lafan1 bvh as our dataset
        self.bvh_path = str(SCRIPT_DIR.parent.parent.parent / "resources" / "lafan1" / "bvh" / "walk1_subject5.bvh")
        self.model_path = os.path.join(ASSETS_PATH, "Model.glb")
        
        # Motion Graph state
        self.mg = None
        self.current_frame = 0
        self.global_transform = Transform.Identity()
        
        # Blending state
        self.is_blending = False
        self.blend_frames = 15
        self.blend_counter = 0
        self.source_frame = 0
        self.target_frame = 0
        self.source_transform = Transform.Identity()
        self.target_transform = Transform.Identity()

    def Start(self):
        print("Loading BVH...")
        self.motion = Motion.LoadFromBVH(self.bvh_path, scale=0.01)
        
        # If motion is too long, we slice it to speed up distance matrix computation
        # For demonstration, 600 frames is enough for a walk cycle graph.
        MAX_FRAMES = 600
        if self.motion.NumFrames > MAX_FRAMES:
            print(f"Slicing motion from {self.motion.NumFrames} to {MAX_FRAMES} frames for demo...")
            self.motion.Frames = self.motion.Frames[:MAX_FRAMES]
            
        self.motion.Debug()

        entity = AI4Animation.Scene.AddEntity("Character")
        self.actor = entity.AddComponent(
            Actor, self.model_path, Definitions.FULL_BODY_NAMES
        )
        
        # Build Motion Graph
        print("Building Motion Graph...")
        dist_mat, transform_mat = compute_distance_matrix(self.motion, window_size=15)
        minima = extract_local_minima(dist_mat, threshold=0.1, local_window=10)
        
        self.mg = MotionGraph(self.motion)
        self.mg.build(dist_mat, transform_mat, minima)
        
        # Initialize at a valid SCC node
        if len(self.mg.scc_nodes) > 0:
            self.current_frame = list(self.mg.scc_nodes)[0]
        else:
            print("WARNING: Graph has no valid SCC! Check distance threshold.")
            self.current_frame = 0

    def Standalone(self):
        AI4Animation.Standalone.Camera.SetTarget(self.actor.Entity)
        AI4Animation.Standalone.Camera.Offset = Vector3.Create(0, 1.5, 3)

    def _get_align_matrix(self, theta, tx, tz):
        # Builds a 4x4 transform matrix from theta, tx, tz
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
        # We process the graph logic at a fixed framerate (e.g. assuming motion is 60fps)
        # We just advance 1 frame per visual Update for simplicity.
        
        if self.is_blending:
            # We are currently in a transition
            w = self.blend_counter / self.blend_frames
            
            src_f = (self.source_frame + self.blend_counter) % self.motion.NumFrames
            tgt_f = (self.target_frame + self.blend_counter) % self.motion.NumFrames
            
            src_poses = self.motion.GetBoneTransformations(timestamps=src_f * self.motion.DeltaTime, bone_names_or_indices=self.actor.GetBoneNames())
            tgt_poses = self.motion.GetBoneTransformations(timestamps=tgt_f * self.motion.DeltaTime, bone_names_or_indices=self.actor.GetBoneNames())
            
            # Apply global transforms
            # Expand global transform to [1, 4, 4] then [num_bones, 4, 4]
            src_g = np.expand_dims(self.source_transform, axis=0)
            tgt_g = np.expand_dims(self.target_transform, axis=0)
            
            src_world = Transform.Multiply(src_g, src_poses)
            tgt_world = Transform.Multiply(tgt_g, tgt_poses)
            
            # Blend
            blended = Transform.Interpolate(src_world, tgt_world, Tensor.Create(w))
            
            self.actor.SetTransforms(blended)
            
            self.blend_counter += 1
            if self.blend_counter >= self.blend_frames:
                self.is_blending = False
                self.current_frame = (self.target_frame + self.blend_frames) % self.motion.NumFrames
                self.global_transform = self.target_transform
        else:
            # Normal playback
            poses = self.motion.GetBoneTransformations(timestamps=self.current_frame * self.motion.DeltaTime, bone_names_or_indices=self.actor.GetBoneNames())
            g = np.expand_dims(self.global_transform, axis=0)
            world_poses = Transform.Multiply(g, poses)
            self.actor.SetTransforms(world_poses)
            
            # Advance logic
            next_frame, trans_type, trans_data = self.mg.get_next_state(self.current_frame)
            
            if trans_type == 'jump':
                print(f"Jumping from {self.current_frame} to {next_frame}")
                self.is_blending = True
                self.blend_counter = 0
                self.source_frame = self.current_frame
                self.target_frame = next_frame
                self.source_transform = self.global_transform
                
                # Apply alignment offset
                theta, tx, tz = trans_data
                align_mat = self._get_align_matrix(theta, tx, tz)
                self.target_transform = Transform.Multiply(self.global_transform, align_mat)
            else:
                self.current_frame = next_frame
                
        self.actor.SyncToScene()
        
        # Auto-screenshot for verification
        if not hasattr(self, '_update_count'):
            self._update_count = 0
        self._update_count += 1
        if self._update_count == 50:
            import raylib as rl
            from ai4animation import Utility
            rl.TakeScreenshot(Utility.ToBytes(str(SCRIPT_DIR / "screenshot_frame50.png")))
            print(f"Screenshot saved at frame {self._update_count}")
        if self._update_count == 150:
            import raylib as rl
            from ai4animation import Utility
            rl.TakeScreenshot(Utility.ToBytes(str(SCRIPT_DIR / "screenshot_frame150.png")))
            print(f"Screenshot saved at frame {self._update_count}")
        
    def GUI(self):
        try:
            import raylib as rl
            rl.TakeScreenshot(b"D:\AnimationTech-learning\raylib_live.png")
        except Exception as e:
            pass
        AI4Animation.Draw.Text(f"Frame: {self.current_frame}", 0.02, 0.02, 0.02, AI4Animation.Color.BLACK)
        state_text = "Blending..." if self.is_blending else "Walking..."
        AI4Animation.Draw.Text(state_text, 0.02, 0.05, 0.02, AI4Animation.Color.BLUE)
        scc_text = f"SCC: {len(self.mg.scc_nodes)} nodes"
        AI4Animation.Draw.Text(scc_text, 0.02, 0.08, 0.02, AI4Animation.Color.GREEN)

if __name__ == "__main__":
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)
