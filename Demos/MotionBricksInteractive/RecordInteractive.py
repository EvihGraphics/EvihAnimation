import os
import sys
import numpy as np
import subprocess
import json

from pathlib import Path
EVIH_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(EVIH_ROOT)

from ai4animation.AI4Animation import AI4Animation
from ai4animation.Math import Vector3, Transform, Quaternion
from ai4animation.Components.MeshRenderer import MeshRenderer
import raylib as rl

class RecordInteractiveApp:
    def __init__(self):
        self.geom_pos = np.load("interactive_geom_pos.npy")
        self.geom_rot = np.load("interactive_geom_rot.npy")
        self.num_frames = self.geom_pos.shape[0]
        
        self.geom_types = np.load("evih_geom_types.npy")
        self.geom_sizes = np.load("evih_geom_sizes.npy")
        self.geom_groups = np.load("evih_geom_groups.npy")
        with open("geom_mesh_names.json", "r") as f:
            self.geom_mesh_names = json.load(f)
        
        self.cam_pos = np.load("interactive_cam_pos.npy")
        self.cam_lookat = np.load("interactive_cam_lookat.npy")
        
        self.current_frame = 0
        self.fps = 30
        self.mesh_entities = []
        
        os.makedirs("evih_mesh_frames", exist_ok=True)
        rl.GetFrameTime = lambda: 1.0 / 30.0
        
    def Start(self):
        AI4Animation.Standalone.Camera.Mode = 4 # Exact Trajectory Match Mode
        self.dummy_target = AI4Animation.Scene.AddEntity("DummyTarget")
        AI4Animation.Standalone.Camera.Camera.fovy = 34.0
        
        rl.rlDisableBackfaceCulling()
        
        with open("geom_colors.json", "r") as f:
            self.geom_colors = json.load(f)
            
        for i, mesh_name in enumerate(self.geom_mesh_names):
            if self.geom_groups[i] != 1:
                self.mesh_entities.append((i, None))
                continue
                
            if mesh_name:
                obj_path = f"meshes_mujoco/Geom_{i}.glb"
                if not os.path.exists(obj_path):
                    self.mesh_entities.append((i, None))
                    continue
                
                model = rl.LoadModel(obj_path.encode('utf-8'))
                entity = AI4Animation.Scene.AddEntity(f"Geom_{i}")
                c = self.geom_colors[i]
                color = (int(c[0]), int(c[1]), int(c[2]), int(c[3]))
                entity.AddComponent(MeshRenderer, model, color)
                self.mesh_entities.append((i, entity))
            else:
                self.mesh_entities.append((i, None))

    def Update(self):
        if self.current_frame >= self.num_frames:
            print("Encoding MP4 with ffmpeg...")
            cmd = ["ffmpeg", "-y", "-framerate", "30", "-i", "evih_mesh_frames/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "interactive_evih_mesh.mp4"]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            AI4Animation.Standalone.Exit()
            return

        for i in range(len(self.mesh_entities)):
            p = self.geom_pos[self.current_frame, i]
            R = self.geom_rot[self.current_frame, i]
            if self.mesh_entities[i][1] is not None:
                idx, entity = self.mesh_entities[i]
                transform = Transform.Identity()
                transform[:3, :3] = R
                transform[:3, 3] = p
                entity.SetTransform(transform)
            
        import pyray as pr
        c = AI4Animation.Standalone.Camera.Camera
        
        c.position.x, c.position.y, c.position.z = self.cam_pos[self.current_frame]
        c.target.x, c.target.y, c.target.z = self.cam_lookat[self.current_frame]
        
        # Up vector is Y in Evih
        c.up.x, c.up.y, c.up.z = 0.0, 1.0, 0.0
        
        c.fovy = 34.0
        
        # Adjust light for shading
        dx = c.target.x - c.position.x
        dy = c.target.y - c.position.y
        dz = c.target.z - c.position.z
        import math
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0.001:
            AI4Animation.Standalone.RenderPipeline.LightDir = pr.Vector3(dx/length, dy/length, dz/length)
            pos = AI4Animation.Standalone.RenderPipeline.LightDir
            AI4Animation.Standalone.RenderPipeline.ShadowLight.position = pr.Vector3(pos.x * -20.0, pos.y * -20.0, pos.z * -20.0)
            AI4Animation.Standalone.RenderPipeline.SunStrength = 2.0

    def Draw(self):
        AI4Animation.Draw.Text(f"Evih Mesh Rendering (Interactive Trace) - Frame {self.current_frame}/{self.num_frames}", 0.05, 0.05, color=AI4Animation.Color.WHITE)

    def GUI(self):
        if self.current_frame < self.num_frames:
            frame_path = f"evih_mesh_frames/frame_{self.current_frame:04d}.png"
            rl.TakeScreenshot(frame_path.encode('utf-8'))
        self.current_frame += 1

if __name__ == "__main__":
    app = RecordInteractiveApp()
    AI4Animation(app)
