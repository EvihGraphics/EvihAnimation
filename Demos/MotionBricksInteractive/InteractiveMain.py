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
from ai4animation.Components.Actor import Actor
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
        
        # Raylib Mesh Entities -> Replaced by Native Evih Actor
        self.Actor = None
        
        # We transform from MuJoCo space (Z-up) to Evih Space (Y-up)
        # using the exact mapping used in the interactive traces to align with the visualizer.
        self.T_m_to_e = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ], dtype=np.float32)

    def Start(self):
        # Load the synthesized skinned GLB via native Evih Actor
        glb_path = os.path.join(os.getcwd(), "g1_skinned.glb")
        if not os.path.exists(glb_path):
            print(f"ERROR: {glb_path} not found. Please run build_g1_skinned.py first.")
            sys.exit(1)
            
        # Get bone names from the GLB or use None to auto-detect
        self.Actor = AI4Animation.Scene.AddEntity("Actor").AddComponent(Actor, glb_path, None, True)
                
        # Initialize inference model state
        self.demo_agent.full_agent.reset()
        
        if self.Actor and self.Actor.Button_Skeleton:
            self.Actor.Button_Skeleton.Active = True
            
        # Fix Camera to follow the robot properly without blend lag
        if self.Actor and len(self.Actor.Entities) > 1:
            def exact_chase_camera_update():
                target_pos = self.Actor.Entities[1].GetPosition()
                cam = AI4Animation.Standalone.Camera.Camera
                # In Raylib struct, we must create a new Vector3 or set fields.
                import pyray
                cam.position = pyray.Vector3(target_pos[0], target_pos[1] + 2.0, target_pos[2] + 3.5)
                cam.target = pyray.Vector3(target_pos[0], target_pos[1] + 1.0, target_pos[2])
                cam.up = pyray.Vector3(0.0, 1.0, 0.0)
            
            AI4Animation.Standalone.Camera.Update = exact_chase_camera_update
            
        # Make the mesh highly visible (actually modify the material color!)
        if hasattr(self.Actor, 'SkinnedMesh') and self.Actor.SkinnedMesh:
            for model in self.Actor.SkinnedMesh.Models:
                model.materials[0].maps[rl.MATERIAL_MAP_ALBEDO].color = pr.BLUE

    def get_raylib_key_states(self):
        # Map Raylib keys to the dictionary expected by WASD_controller
        w_pressed = rl.IsKeyDown(rl.KEY_W)
        a_pressed = rl.IsKeyDown(rl.KEY_A)
        
        if hasattr(self, 'args') and self.args.auto_record:
            step = getattr(self, 'current_step', 0)
            w_pressed = (50 <= step < 300)
            a_pressed = (150 <= step < 250)

        return {
            "w": w_pressed, "a": a_pressed,
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

        # 5. Forward Kinematics
        mujoco.mj_kinematics(self.demo_agent.mj_model, self.demo_agent.mj_data)
        
        # 6. Apply to Native Actor
        if self.Actor:
            num_bones = self.Actor.GetBoneCount()
            transforms = Transform.Identity(num_bones)
            
            for i in range(min(num_bones, self.demo_agent.mj_model.nbody)):
                p_m = self.demo_agent.mj_data.xpos[i]
                R_m = self.demo_agent.mj_data.xmat[i].reshape(3, 3)
                
                # Transform to Evih space
                p_e = self.T_m_to_e @ p_m
                R_e = self.T_m_to_e @ R_m @ self.T_m_to_e.T
                
                # Build native transform
                transforms[i] = Transform.TR(p_e, R_e)
                
            # Evih camera follows the root
            root_pos_e = transforms[0][:3, 3] # Pelvis position
            # Add offset
            cam_pos_e = root_pos_e + np.array([0.0, 2.0, 3.5])
            cam_lookat_e = root_pos_e + np.array([0.0, 1.0, 0.0])
            
            # Convert to PyRay vectors
            cam_pos_pr = pr.Vector3(float(cam_pos_e[0]), float(cam_pos_e[1]), float(cam_pos_e[2]))
            cam_lookat_pr = pr.Vector3(float(cam_lookat_e[0]), float(cam_lookat_e[1]), float(cam_lookat_e[2]))
            
            AI4Animation.Standalone.Camera.Camera.position = cam_pos_pr
            AI4Animation.Standalone.Camera.Camera.target = cam_lookat_pr
                
            self.Actor.SetTransforms(transforms)
        
        if hasattr(self, 'args') and self.args.auto_record:
            if not hasattr(self, 'qpos_history'):
                self.qpos_history = []
            self.qpos_history.append(qpos.copy())

        if self.args.auto_record:
            os.makedirs("evih_headless_frames", exist_ok=True)
            rl.TakeScreenshot(bytes(f"evih_headless_frames/frame_{self.current_step:04d}.png", "utf-8"))
            if self.current_step >= self.args.max_steps:
                np.save("interactive_qpos_latest.npy", np.array(self.qpos_history))
                AI4Animation.Standalone.Exit()

    def Draw(self):
        AI4Animation.Draw.Text("MotionBricks Native Integration [WASD Control]", 0.05, 0.05, color=AI4Animation.Color.BLACK)
        AI4Animation.Draw.Text("EvihAnimation Actor rendering Skinned G1 Mesh", 0.05, 0.10, color=AI4Animation.Color.BLACK)

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
    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")
    parser.add_argument("--auto_record", action="store_true", help="Auto record headless video")
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
