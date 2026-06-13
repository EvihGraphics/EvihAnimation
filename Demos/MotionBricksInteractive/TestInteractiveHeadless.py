import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import argparse
import numpy as np

# We bypass Raylib by mocking it BEFORE we import InteractiveMain
import sys
import types

mock_rl = types.ModuleType("raylib")
mock_rl.rlDisableBackfaceCulling = lambda: None
mock_rl.IsKeyDown = lambda key: False
mock_rl.KEY_W = 1
mock_rl.KEY_A = 2
mock_rl.KEY_S = 3
mock_rl.KEY_D = 4
mock_rl.KEY_LEFT = 5
mock_rl.KEY_RIGHT = 6
mock_rl.KEY_UP = 7
mock_rl.KEY_DOWN = 8
mock_rl.KEY_Z = 9
mock_rl.KEY_X = 10
mock_rl.KEY_C = 11
mock_rl.KEY_V = 12
mock_rl.KEY_B = 13
mock_rl.KEY_R = 14
mock_rl.KEY_T = 15
mock_rl.KEY_F = 16
mock_rl.KEY_G = 17
mock_rl.KEY_Q = 18
mock_rl.KEY_E = 19
mock_rl.KEY_LEFT_SHIFT = 20
mock_rl.KEY_RIGHT_SHIFT = 21
mock_rl.KEY_LEFT_CONTROL = 22
mock_rl.KEY_RIGHT_CONTROL = 23
mock_rl.KEY_ENTER = 24
mock_rl.LoadModel = lambda path: None
mock_rl.rl = mock_rl
mock_rl.ffi = mock_rl
sys.modules["raylib"] = mock_rl

mock_pyray = types.ModuleType("pyray")
class MockVector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x; self.y = y; self.z = z
mock_pyray.Vector3 = MockVector3
sys.modules["pyray"] = mock_pyray

# We also mock AI4Animation so it doesn't try to load GL contexts
class MockCamera:
    def __init__(self):
        class Vec3:
            def __init__(self):
                self.x, self.y, self.z = 0,0,0
        self.position = Vec3()
        self.target = Vec3()
        self.up = Vec3()

class MockStandalone:
    def __init__(self):
        class Cam:
            def __init__(self):
                self.Camera = MockCamera()
                self.Mode = 0
        self.Camera = Cam()
    def Exit(self):
        sys.exit(0)

class MockScene:
    def AddEntity(self, name):
        class Ent:
            def AddComponent(self, *args, **kwargs): pass
            def SetTransform(self, t): pass
        return Ent()

class MockAI4Animation:
    Standalone = MockStandalone()
    Scene = MockScene()
    def __init__(self, *args, **kwargs): pass

mock_ai4 = types.ModuleType("ai4animation")
sys.modules["ai4animation.AI4Animation"] = mock_ai4
sys.modules["ai4animation.AI4Animation"].AI4Animation = MockAI4Animation

mock_components = types.ModuleType("ai4animation.Components")
mock_mr = types.ModuleType("ai4animation.Components.MeshRenderer")
mock_mr.MeshRenderer = type("MeshRenderer", (), {})
sys.modules["ai4animation.Components.MeshRenderer"] = mock_mr

import InteractiveMain
from InteractiveMain import InteractiveApp

def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanoid_xml", type=str, default=f"{InteractiveMain.MOTIONBRICKS_ROOT_SYS}/assets/skeletons/g1/scene_29dof.xml")
    parser.add_argument("--result_dir", type=str, default=f"{InteractiveMain.MOTIONBRICKS_ROOT_SYS}/out")
    parser.add_argument("--data_root", type=str, default=f"{InteractiveMain.MOTIONBRICKS_ROOT_SYS}/datasets")
    parser.add_argument("--explicit_dataset_folder", type=str, default=None)
    parser.add_argument("--reprocess_clips", type=int, default=0)
    parser.add_argument("--controller", type=str, default="wasd")
    parser.add_argument("--lookat_movement_direction", type=int, default=0)
    parser.add_argument("--has_viewer", type=int, default=0)
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--random_speed_scale", type=int, default=0)
    parser.add_argument("--speed_scale", type=str, default="0.8,1.2")
    parser.add_argument("--generate_dt", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")
    args = parser.parse_args([])
    
    args.return_model_configs = True
    args.return_dataloader = True
    args.recording_dir = None
    args.EXP = args.planner
    args.speed_scale = [float(i) for i in args.speed_scale.split(",")]

    print("Initializing App...")
    app = InteractiveApp(args)
    app.Start()
    
    import mujoco
    import cv2
    import subprocess
    
    renderer = mujoco.Renderer(app.demo_agent.mj_model, 480, 640)
    
    # We will simulate WASD inputs:
    # 0-50: Stand idle
    # 50-100: Press W
    # 100-150: Press W + A
    # 150-200: Press W
    print("Testing 200 frames of inference and recording video...")
    
    os.makedirs("headless_frames", exist_ok=True)
    
    for i in range(200):
        # Simulate WASD inputs
        w_pressed = (50 <= i < 200)
        a_pressed = (100 <= i < 150)
        
        mock_rl.IsKeyDown = lambda key: (key == mock_rl.KEY_W and w_pressed) or (key == mock_rl.KEY_A and a_pressed)
            
        app.Update()
        
        # Render the MuJoCo scene using offscreen renderer
        mujoco.mj_forward(app.demo_agent.mj_model, app.demo_agent.mj_data)
        
        # Update camera to track pelvis (like in InteractiveMain)
        pelvis_pos_m = app.demo_agent.mj_data.subtree_com[1]
        
        cam = mujoco.MjvCamera()
        cam.distance = 3.0
        cam.elevation = -20
        cam.azimuth = 90
        cam.lookat[:] = pelvis_pos_m
        
        renderer.update_scene(app.demo_agent.mj_data, camera=cam)
        pixels = renderer.render()
        
        # pixels is RGB, cv2 needs BGR
        pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"headless_frames/frame_{i:04d}.png", pixels_bgr)
        
        qpos = app.demo_agent.mj_data.qpos
        if i % 10 == 0:
            print(f"Step {i}: Pelvis Y: {qpos[1]:.4f}, Pelvis Z: {qpos[2]:.4f}")
            
    print("Encoding video...")
    cmd = ["ffmpeg", "-y", "-framerate", "30", "-i", "headless_frames/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "interactive_headless_test.mp4"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("SUCCESS: Headless integration test video generated perfectly!")

if __name__ == "__main__":
    run_test()
