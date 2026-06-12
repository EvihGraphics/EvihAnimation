import os
import sys
import argparse
from pathlib import Path
import numpy as np
import subprocess
import json

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ai4animation.AI4Animation import AI4Animation
from ai4animation.Math import Vector3, Transform, Quaternion
from ai4animation.Components.MeshRenderer import MeshRenderer
import raylib as rl

class DummyTarget:
    def __init__(self):
        self.transform = Transform.Identity()
    def GetTransform(self):
        return self.transform
    def GetPosition(self):
        return Transform.GetPosition(self.transform)

class MotionBricksReplayApp:
    def __init__(self, positions_file, parents_file, auto_record=False):
        self.positions = np.load(positions_file)
        self.parents = np.load(parents_file)
        self.num_frames = self.positions.shape[0]
        self.num_joints = self.positions.shape[1]
        
        self.geom_pos = np.load("evih_geom_pos.npy")
        self.geom_rot = np.load("evih_geom_rot.npy")
        with open("geom_mesh_names.json", "r") as f:
            self.geom_mesh_names = json.load(f)
            
        self.current_frame = 0
        self.fps = 30
        self.playing = True
        self.auto_record = auto_record
        self.dummy_target = DummyTarget()
        self.mesh_entities = []
        
        if self.auto_record:
            os.makedirs("frames", exist_ok=True)
            rl.GetFrameTime = lambda: 1.0 / 30.0
        
    def Start(self):
        AI4Animation.Standalone.Camera.Mode = 2 # Third Person
        AI4Animation.Standalone.Camera.SetTarget(self.dummy_target)
        
        for i, mesh_name in enumerate(self.geom_mesh_names):
            if mesh_name:
                obj_path = f"meshes/{mesh_name}.obj"
                if os.path.exists(obj_path):
                    model = rl.LoadModel(obj_path.encode('utf-8'))
                    entity = AI4Animation.Scene.AddEntity(f"Geom_{i}")
                    entity.AddComponent(MeshRenderer, model)
                    self.mesh_entities.append((i, entity))

    def Update(self):
        if self.playing:
            self.current_frame += 1
            if self.current_frame >= self.num_frames:
                self.current_frame = 0
                if self.auto_record:
                    print("Encoding MP4 with ffmpeg...")
                    cmd = ["ffmpeg", "-y", "-framerate", "30", "-i", "frames/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "evihanimation_true_mesh_replay.mp4"]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    AI4Animation.Standalone.Exit()

        root_pos = self.positions[self.current_frame, 1]
        self.dummy_target.transform[0:3, 3] = root_pos
        
        # Update Mesh entities
        for geom_idx, entity in self.mesh_entities:
            p = self.geom_pos[self.current_frame, geom_idx]
            R = self.geom_rot[self.current_frame, geom_idx]
            q = Quaternion.FromMatrix(R)
            
            # Construct a Transform matrix manually or use SetPosition/SetRotation
            transform = Transform.Identity()
            transform[:3, :3] = R
            transform[3, :3] = p
            entity.SetTransform(transform)

    def Draw(self):
        positions = self.positions[self.current_frame]
        for i in range(1, self.num_joints):
            parent = self.parents[i]
            if parent == 0:
                continue
            p1 = positions[i]
            p2 = positions[parent]
            AI4Animation.Draw.Cylinder(p1, p2, 0.01, 0.01, color=AI4Animation.Color.GREEN)
            AI4Animation.Draw.Sphere(p1, 0.015, color=AI4Animation.Color.RED)
            
        AI4Animation.Draw.Text(f"Frame: {self.current_frame} / {self.num_frames} (G1 Mesh)", 0.05, 0.05, color=AI4Animation.Color.WHITE)

    def GUI(self):
        if self.auto_record:
            frame_path = f"frames/frame_{self.current_frame:04d}.png"
            rl.TakeScreenshot(frame_path.encode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-record", action="store_true")
    args = parser.parse_args()
    
    app = MotionBricksReplayApp("evih_positions.npy", "parents.npy", auto_record=args.auto_record)
    AI4Animation(app)



