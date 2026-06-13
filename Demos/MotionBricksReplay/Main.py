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
    def __init__(self, positions_file, parents_file, auto_record=False, export_mask=False):
        self.export_mask = export_mask
        self.positions = np.load(positions_file)
        self.parents = np.load(parents_file)
        self.num_frames = self.positions.shape[0]
        self.num_joints = self.positions.shape[1]
        
        self.geom_pos = np.load("evih_geom_pos.npy")
        self.geom_rot = np.load("evih_geom_rot.npy")
        self.geom_types = np.load("evih_geom_types.npy")
        self.geom_sizes = np.load("evih_geom_sizes.npy")
        self.geom_groups = np.load("evih_geom_groups.npy")
        with open("geom_mesh_names.json", "r") as f:
            self.geom_mesh_names = json.load(f)
        
        self.cam_pos = np.load("evih_cam_pos.npy")
        self.cam_lookat = np.load("evih_cam_lookat.npy")
        self.cam_up = np.load("evih_cam_up.npy")
        
        import pyray as pr
        self.cam_pos_v = [pr.Vector3(*p) for p in self.cam_pos]
        self.cam_lookat_v = [pr.Vector3(*p) for p in self.cam_lookat]
        self.cam_up_v = [pr.Vector3(*p) for p in self.cam_up]
        
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
        AI4Animation.Standalone.Camera.Mode = 4 # Exact Trajectory Match Mode
        
        # We don't need dummy_target if we enforce exact camera position
        self.dummy_target = AI4Animation.Scene.AddEntity("DummyTarget")
        AI4Animation.Standalone.Camera.Camera.fovy = 34.0
        
        # Disable backface culling to ensure all meshes render fully
        rl.rlDisableBackfaceCulling()
        
        import json
        with open("geom_colors.json", "r") as f:
            self.geom_colors = json.load(f)
            
        for i, mesh_name in enumerate(self.geom_mesh_names):
            if self.geom_groups[i] != 1:
                self.mesh_entities.append((i, None))
                continue
                
            if mesh_name:
                obj_path = f"meshes_mujoco/Geom_{i}.glb"
                if not os.path.exists(obj_path):
                    print(f"Warning: Mesh {obj_path} not found.")
                    continue
                
                model = rl.LoadModel(obj_path.encode('utf-8'))
                entity = AI4Animation.Scene.AddEntity(f"Geom_{i}")
                c = self.geom_colors[i]
                color = (int(c[0]), int(c[1]), int(c[2]), int(c[3]))
                if getattr(self, 'export_mask', False):
                    color = (0, 0, 0, 255)
                entity.AddComponent(MeshRenderer, model, color)
                self.mesh_entities.append((i, entity))
            else:
                self.mesh_entities.append((i, None))

    def Update(self):
        if self.playing:
            self.current_frame += 1
            if self.current_frame >= self.num_frames:
                self.current_frame = 0
                if self.auto_record:
                    print("Encoding MP4 with ffmpeg...")
                    video_name = getattr(self, 'video_name', 'evihanimation_true_mesh_replay.mp4')
                    cmd = ["ffmpeg", "-y", "-framerate", "30", "-i", "frames/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", video_name]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    AI4Animation.Standalone.Exit()

        # Dummy target tracking removed as camera is manually enforced
        # Update Mesh entities
        for i in range(len(self.mesh_entities)):
            p = self.geom_pos[self.current_frame, i]
            R = self.geom_rot[self.current_frame, i]
            if self.mesh_entities[i][1] is not None:
                # Mesh entity update
                idx, entity = self.mesh_entities[i]
                transform = Transform.Identity()
                transform[:3, :3] = R
                transform[:3, 3] = p
                entity.SetTransform(transform)
            
        # Update Camera
        frame_idx = self.current_frame
        if frame_idx < len(self.cam_pos):
            import pyray as pr
            cam = AI4Animation.Standalone.Camera.Camera
            
            # Headlight: light points from camera to target to ensure front of robot is fully illuminated
            import math
            dx = self.cam_lookat[frame_idx][0] - self.cam_pos[frame_idx][0]
            dy = self.cam_lookat[frame_idx][1] - self.cam_pos[frame_idx][1]
            dz = self.cam_lookat[frame_idx][2] - self.cam_pos[frame_idx][2]
            length = math.sqrt(dx*dx + dy*dy + dz*dz)
            if length > 0.001 and not getattr(self, 'export_mask_mode', False):
                AI4Animation.Standalone.RenderPipeline.LightDir = pr.Vector3(dx/length, dy/length, dz/length)
                pos = AI4Animation.Standalone.RenderPipeline.LightDir
                AI4Animation.Standalone.RenderPipeline.ShadowLight.position = pr.Vector3(pos.x * -20.0, pos.y * -20.0, pos.z * -20.0)
                AI4Animation.Standalone.RenderPipeline.SunStrength = 2.0

        # Enforce exact MuJoCo camera trajectory
        c = AI4Animation.Standalone.Camera.Camera
        c.position.x, c.position.y, c.position.z = self.cam_pos[self.current_frame]
        c.target.x, c.target.y, c.target.z = self.cam_lookat[self.current_frame]
        c.up.x, c.up.y, c.up.z = self.cam_up[self.current_frame]
        c.fovy = 34.0
        
        if self.current_frame == 50:
            head_pos_v = pr.Vector3(*np.load('evih_geom_pos.npy')[self.current_frame, 40])
            screen_pos = rl.GetWorldToScreen(head_pos_v, c)
            print(f'EVIH FRAME 50 HEAD Y: {screen_pos.y} X: {screen_pos.x}')

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
            
        # Draw the primitive geoms that don't have meshes
        for i in range(len(self.mesh_entities)):
            if self.geom_groups[i] != 1:
                continue
            if self.mesh_entities[i][1] is None:
                p = self.geom_pos[self.current_frame, i]
                R = self.geom_rot[self.current_frame, i]
                c = self.geom_colors[i]
                color = (int(c[0]), int(c[1]), int(c[2]), int(c[3]))
                
                if getattr(self, 'export_mask', False):
                    color = AI4Animation.Color.BLACK
                
                if self.geom_types[i] == 5:
                    # Cylinders (visual shoulders/elbows)
                    radius = self.geom_sizes[i, 0]
                    half_length = self.geom_sizes[i, 1]
                    local_axis = R[:, 2] # NOTE: MuJoCo cylinders are oriented along the local Z axis!
                    p1 = p - local_axis * half_length
                    p2 = p + local_axis * half_length
                    AI4Animation.Draw.Cylinder(p1, p2, radius, radius, color=color)
                elif self.geom_types[i] == 2:
                    # Spheres (visual feet components)
                    radius = self.geom_sizes[i, 0]
                    AI4Animation.Draw.Sphere(p, size=radius, color=color)
            
        if not getattr(self, 'export_mask_mode', False):
            AI4Animation.Draw.Text(f"Frame: {self.current_frame} / {self.num_frames} (G1 Mesh)", 0.05, 0.05, color=AI4Animation.Color.WHITE)

    def GUI(self):
        if self.auto_record:
            frame_path = f"frames/frame_{self.current_frame:04d}.png"
            rl.TakeScreenshot(frame_path.encode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-record", action="store_true")
    parser.add_argument("--export-mask", action="store_true")
    args = parser.parse_args()
    
    app = MotionBricksReplayApp("evih_positions.npy", "parents.npy", auto_record=args.auto_record, export_mask=args.export_mask)
    
    if args.export_mask:
        app.export_mask_mode = True
            
        # We need to hide the grid and background
        def hide_grid_and_set_bg():
            rl.rlDisableBackfaceCulling()
            for entity in AI4Animation.Scene.Entities:
                if entity.Name.startswith("Ground") or entity.Name.startswith("Wall"):
                    # Try to remove grid or move entity away
                    t = Transform.Identity()
                    t[1, 3] = -1000
                    entity.SetTransform(t)
            AI4Animation.Standalone.RenderPipeline.SunStrength = 0.0
            AI4Animation.Standalone.RenderPipeline.AmbientStrength = 1.0
            try:
                AI4Animation.Standalone.RenderPipeline.SkyColor.x = 1.0
                AI4Animation.Standalone.RenderPipeline.SkyColor.y = 1.0
                AI4Animation.Standalone.RenderPipeline.SkyColor.z = 1.0
            except:
                pass
            
        # Hook into Start
        original_start = app.Start
        def new_start():
            original_start()
            hide_grid_and_set_bg()
            # Override colors to pure black for the mask
            for i in range(len(app.geom_colors)):
                app.geom_colors[i] = [0, 0, 0, 255]
            for i, entity in app.mesh_entities:
                if entity:
                    entity.GetComponent(MeshRenderer).Color = (0, 0, 0, 255)
        app.Start = new_start
        
        # Override output video name
        app.video_name = "evihanimation_mask_replay.mp4"
    else:
        app.video_name = "evihanimation_true_mesh_replay.mp4"

    AI4Animation(app)



