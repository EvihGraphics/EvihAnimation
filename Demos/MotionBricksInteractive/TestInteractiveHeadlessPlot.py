import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# We bypass Raylib by mocking it BEFORE we import InteractiveMain
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
for k in "ZXCVBRTFGQE":
    setattr(mock_rl, f"KEY_{k}", ord(k))
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
        self.position = MockVector3()
        self.target = MockVector3()
        self.up = MockVector3()

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
    parser.add_argument("--max_steps", type=int, default=300)
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
    
    # We will simulate WASD inputs:
    # 0-50: Stand idle
    # 50-150: Press W
    # 150-250: Press W + A (Turn left)
    # 250-300: Press W
    
    print("Testing 300 frames of inference and recording plot video...")
    
    positions = []
    
    for i in range(300):
        w_pressed = (50 <= i < 300)
        a_pressed = (150 <= i < 250)
        
        mock_rl.IsKeyDown = lambda key: (key == mock_rl.KEY_W and w_pressed) or (key == mock_rl.KEY_A and a_pressed)
            
        app.Update()
        
        pelvis_pos = app.demo_agent.mj_data.qpos[:3].copy()
        positions.append(pelvis_pos)
        
    positions = np.array(positions)
    
    # Create Matplotlib Animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Robot Pelvis Trajectory (Top-Down)')
    
    line, = ax.plot([], [], 'b-', alpha=0.5, label='Path')
    point, = ax.plot([], [], 'ro', label='Robot')
    
    # Draw path segments colored by action
    idle_path, = ax.plot(positions[:50, 0], positions[:50, 1], 'gray', label='Idle (0-50)')
    w_path, = ax.plot(positions[50:150, 0], positions[50:150, 1], 'g-', label='Forward W (50-150)')
    wa_path, = ax.plot(positions[150:250, 0], positions[150:250, 1], 'orange', label='Turn Left W+A (150-250)')
    w2_path, = ax.plot(positions[250:, 0], positions[250:, 1], 'g--', label='Forward W (250-300)')
    
    ax.legend(loc='upper left')

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    os.makedirs("plot_frames", exist_ok=True)
    for i in range(300):
        line.set_data(positions[:i, 0], positions[:i, 1])
        point.set_data([positions[i, 0]], [positions[i, 1]])
        fig.savefig(f"plot_frames/frame_{i:04d}.png")
        
    print("SUCCESS: PNG frames generated in plot_frames directory.")

if __name__ == "__main__":
    run_test()
