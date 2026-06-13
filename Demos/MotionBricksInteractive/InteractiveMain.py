import os
import sys
import argparse
from pathlib import Path
import numpy as np
import time

# EvihAnimation Root
EVIH_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(EVIH_ROOT)

# MotionBricks Path
import platform
if platform.system() == 'Windows':
    MOTIONBRICKS_ROOT_SYS = r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks'
else:
    MOTIONBRICKS_ROOT_SYS = '/root/Project/GR00T-WholeBodyControl/motionbricks'

if MOTIONBRICKS_ROOT_SYS not in sys.path:
    sys.path.append(MOTIONBRICKS_ROOT_SYS)

from ai4animation.AI4Animation import AI4Animation
from ai4animation.Math import Vector3, Transform, Quaternion
from ai4animation.Components.MeshRenderer import MeshRenderer
import raylib as rl
import pyray as pr
import mujoco

from types import SimpleNamespace

# MotionBricks Imports
from motionbricks.motion_backbone.demo.utils import navigation_demo

class InteractiveApp:
    def __init__(self, args):
        self.args = args
        self.fps = 30
        
        # Initialize MotionBricks Demo Agent
        # Change working directory temporarily to MOTIONBRICKS_ROOT because
        # hydra configs inside the ckpt use relative paths (e.g. 'out/...')
        original_cwd = os.getcwd()
        os.chdir(MOTIONBRICKS_ROOT_SYS)
        try:
            self.demo_agent = navigation_demo(args)
        finally:
            os.chdir(original_cwd)
        
        # Raylib Mesh Entities
        self.mesh_entities = []
        
        # Load Evih mesh metadata (shared from replay demo)
        self.geom_types = np.load("evih_geom_types.npy")
        self.geom_sizes = np.load("evih_geom_sizes.npy")
        self.geom_groups = np.load("evih_geom_groups.npy")
        import json
        with open("geom_mesh_names.json", "r") as f:
            self.geom_mesh_names = json.load(f)
        with open("geom_colors.json", "r") as f:
            self.geom_colors = json.load(f)
            
        self.T_m_to_e = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ], dtype=np.float32)

    def Start(self):
        AI4Animation.Standalone.Camera.Mode = 4 # Exact Trajectory Match Mode
        AI4Animation.Standalone.Camera.Camera.fovy = 34.0
        
        # Disable backface culling
        rl.rlDisableBackfaceCulling()
        
        for i, mesh_name in enumerate(self.geom_mesh_names):
            if self.geom_groups[i] != 1:
                self.mesh_entities.append((i, None))
                continue
                
            if mesh_name:
                obj_path = f"meshes_mujoco/Geom_{i}.glb"
                if not os.path.exists(obj_path):
                    print(f"Warning: Mesh {obj_path} not found.")
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
                
        # Initialize inference model state
        self.demo_agent.full_agent.reset()

    def get_raylib_key_states(self):
        # Map Raylib keys to the dictionary expected by WASD_controller
        return {
            "w": rl.IsKeyDown(rl.KEY_W), "a": rl.IsKeyDown(rl.KEY_A),
            "s": rl.IsKeyDown(rl.KEY_S), "d": rl.IsKeyDown(rl.KEY_D),
            "left": rl.IsKeyDown(rl.KEY_LEFT), "right": rl.IsKeyDown(rl.KEY_RIGHT),
            "up": rl.IsKeyDown(rl.KEY_UP), "down": rl.IsKeyDown(rl.KEY_DOWN),
            "z": rl.IsKeyDown(rl.KEY_Z), "x": rl.IsKeyDown(rl.KEY_X),
            "c": rl.IsKeyDown(rl.KEY_C), "v": rl.IsKeyDown(rl.KEY_V),
            "b": rl.IsKeyDown(rl.KEY_B), "r": rl.IsKeyDown(rl.KEY_R),
            "t": rl.IsKeyDown(rl.KEY_T), "f": rl.IsKeyDown(rl.KEY_F),
            "g": rl.IsKeyDown(rl.KEY_G), "q": rl.IsKeyDown(rl.KEY_Q),
            "e": rl.IsKeyDown(rl.KEY_E),
            "shift": rl.IsKeyDown(rl.KEY_LEFT_SHIFT) or rl.IsKeyDown(rl.KEY_RIGHT_SHIFT),
            "ctrl": rl.IsKeyDown(rl.KEY_LEFT_CONTROL) or rl.IsKeyDown(rl.KEY_RIGHT_CONTROL),
            "enter": rl.IsKeyDown(rl.KEY_ENTER)
        }

    def Update(self):
        step_start = time.time()
        
        if not hasattr(self, 'current_step'):
            self.current_step = 0
        self.current_step += 1
        
        if self.current_step > self.args.max_steps:
            AI4Animation.Standalone.Exit()
            return

        # 1. Fetch next frame from Agent
        qpos = self.demo_agent.full_agent.get_next_frame()
        context_motion_features = self.demo_agent.full_agent.get_context_motion_features()
        context_mujoco_qpos = self.demo_agent.full_agent.get_context_mujoco_qpos()
        self.demo_agent.mj_data.qpos[:] = qpos

        # 2. Capture user inputs via Raylib
        key_pressed = self.get_raylib_key_states()
        
        # We need a viewer-like object to supply camera parameters to the controller.
        # Evih's camera provides these parameters:
        class DummyViewerCam:
            def __init__(self, cam):
                # Convert Evih position back to MuJoCo space for the controller!
                # T_m_to_e: (x, y, z) -> (y, z, x). So T_e_to_m: (x, y, z) -> (z, x, y)
                self.lookat = np.array([cam.target.z, cam.target.x, cam.target.y])
                pos = np.array([cam.position.z, cam.position.x, cam.position.y])
                diff = pos - self.lookat
                self.distance = np.linalg.norm(diff)
                # azimuth and elevation in radians.
                # In MuJoCo: x is forward, y is left, z is up.
                self.elevation = -np.arcsin(diff[2] / (self.distance + 1e-5)) * 180.0 / np.pi
                self.azimuth = -np.arctan2(diff[0], diff[1]) * 180.0 / np.pi

        class DummyViewer:
            def __init__(self, cam):
                self.cam = DummyViewerCam(cam)
                
        viewer = DummyViewer(AI4Animation.Standalone.Camera.Camera)

        # 3. Generate control signals
        control_signals = self.demo_agent.controller.generate_control_signals(
            viewer, self.demo_agent.mj_model, self.demo_agent.mj_data, visualize=False,
            control_info={
                "force_idle": False, 
                "allowed_mode": getattr(self.args, 'allowed_mode', None),
                "key_pressed": key_pressed
            }
        )
        if self.args.use_qpos:
            control_signals['context_mujoco_qpos'] = context_mujoco_qpos
        else:
            control_signals['context_motion_features'] = context_motion_features

        # 4. Neural Network Inference for future frames
        import torch as t
        with t.no_grad():
            self.demo_agent.full_agent.generate_new_frames(
                control_signals,
                self.demo_agent.controller.get_controller_dt() * self.args.generate_dt
            )

        # 5. Forward Kinematics to update meshes
        mujoco.mj_forward(self.demo_agent.mj_model, self.demo_agent.mj_data)
        
        for i in range(len(self.mesh_entities)):
            if self.mesh_entities[i][1] is not None:
                p_m = self.demo_agent.mj_data.geom_xpos[i]
                R_m = self.demo_agent.mj_data.geom_xmat[i].reshape(3, 3)
                
                # Transform to Evih space
                p_e = self.T_m_to_e @ p_m
                R_e = self.T_m_to_e @ R_m @ self.T_m_to_e.T
                
                idx, entity = self.mesh_entities[i]
                transform = Transform.Identity()
                transform[:3, :3] = R_e
                transform[:3, 3] = p_e
                entity.SetTransform(transform)

        # Chase Camera Logic: Track the Pelvis
        # We can extract the Pelvis from prev_qpos (since the controller caches it)
        # or from d.subtree_com[1]
        pelvis_pos_m = self.demo_agent.mj_data.subtree_com[1]
        pelvis_pos_e = self.T_m_to_e @ pelvis_pos_m
        
        # Basic Chase Cam
        cam = AI4Animation.Standalone.Camera.Camera
        cam.target.x, cam.target.y, cam.target.z = pelvis_pos_e
        
        # Maintain offset relative to target
        # e.g. distance = 3, slight elevation
        cam.position.x = cam.target.x + 3.0
        cam.position.y = cam.target.y + 1.5
        cam.position.z = cam.target.z

    def Draw(self):
        AI4Animation.Draw.Text("MotionBricks Interactive Demo [WASD Control]", 0.05, 0.05, color=AI4Animation.Color.WHITE)
        AI4Animation.Draw.Text("Green: Evih True-Mesh (PyTorch Inference)", 0.05, 0.10, color=AI4Animation.Color.GREEN)

    def GUI(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive demo for the G1 humanoid inside EvihAnimation")
    
    # Path configs
    import platform
    if platform.system() == 'Windows':
        MOTIONBRICKS_ROOT = r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks'
    else:
        MOTIONBRICKS_ROOT = '/root/Project/GR00T-WholeBodyControl/motionbricks'
        
    parser.add_argument("--humanoid_xml", type=str, default=f"{MOTIONBRICKS_ROOT}/assets/skeletons/g1/scene_29dof.xml")
    parser.add_argument("--result_dir", type=str, default=f"{MOTIONBRICKS_ROOT}/out")
    parser.add_argument("--data_root", type=str, default=f"{MOTIONBRICKS_ROOT}/datasets")
    parser.add_argument("--explicit_dataset_folder", type=str, default=None)
    parser.add_argument("--reprocess_clips", type=int, default=0)

    # Controller config
    parser.add_argument("--controller", type=str, default="wasd", choices=["wasd", "random"])
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

    # Run configs
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--num_runs", type=int, default=1)

    # Model configurations
    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")

    args = parser.parse_args()

    args.return_model_configs = True
    args.return_dataloader = True
    args.recording_dir = None
    args.EXP = args.planner
    args.speed_scale = [float(i) for i in args.speed_scale.split(",")]
    
    import torch as t
    random_seed = args.random_seed
    np.random.seed(random_seed)
    t.manual_seed(random_seed)

    # Need to pass args to AI4Animation setup eventually, but for now we run AI4Animation instance
    app = InteractiveApp(args)
    AI4Animation(app)
