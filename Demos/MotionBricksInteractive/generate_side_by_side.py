import os
import sys
import numpy as np
import mujoco
from PIL import Image

# Load recorded trajectory from InteractiveMain
try:
    qpos_history = np.load("interactive_qpos_latest.npy")
except:
    qpos_history = np.load("interactive_qpos.npy")
num_frames = min(150, qpos_history.shape[0])

# Fix for linux vs windows
import platform
if platform.system() == 'Windows':
    MOTIONBRICKS_ROOT = r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks'
else:
    MOTIONBRICKS_ROOT = '/root/Project/GR00T-WholeBodyControl/motionbricks'

model_path = f"{MOTIONBRICKS_ROOT}/assets/skeletons/g1/scene_29dof.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

# Evih uses Y-up, MuJoCo uses Z-up. Wait, we want to match Evih's camera!
# In InteractiveMain.py, Camera setup:
# self.Camera.SetPosition(np.array([0.0, 1.5, 3.0]))
# self.Camera.SetRotation(Rotation.RotationX(-30)) -> wait, fovy=34.0?
# Actually, the user wants to see them side-by-side. 
# I will just use standard MuJoCo Free Camera matching Evih.
renderer = mujoco.Renderer(m, 480, 640)

# Setup MuJoCo camera
m.opt.timestep = 0.005

os.makedirs("side_by_side_frames", exist_ok=True)

T_m_to_e = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
], dtype=np.float32)
T_e_to_m = T_m_to_e.T

for step in range(1, num_frames + 1):
    d.qpos[:] = qpos_history[step-1]
    mujoco.mj_forward(m, d)
    
    # Pelvis position in MuJoCo must perfectly match the bone transform
    pelvis_m = d.xpos[1]
    pelvis_e = T_m_to_e @ pelvis_m
    
    # Evih Camera Mode 1:
    # position = self.Target.GetPosition() + Vector3.Create(0.0, 2.0, self.Distance)
    # target = self.Target.GetPosition() + Vector3.Create(0.0, 1.0, 0.0)
    cam_pos_e = pelvis_e + np.array([0.0, 2.0, 3.5])
    cam_lookat_e = pelvis_e + np.array([0.0, 1.0, 0.0])
    
    # Transform back to MuJoCo Space for the free camera
    cam_pos_m = T_e_to_m @ cam_pos_e
    cam_lookat_m = T_e_to_m @ cam_lookat_e
    
    # We need a custom camera to render with exact pos/lookat
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    
    # Calculate elevation and azimuth from pos and lookat
    diff = cam_pos_m - cam_lookat_m
    distance = np.linalg.norm(diff)
    elevation = -np.arcsin(diff[2] / (distance + 1e-5)) * 180.0 / np.pi
    azimuth = np.arctan2(diff[0], -diff[1]) * 180.0 / np.pi
    
    cam.lookat[:] = cam_lookat_m
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth
    
    renderer.update_scene(d, camera=cam)
    mj_img = renderer.render()
    mj_pil = Image.fromarray(mj_img)
    
    # Render segmentation mask
    renderer.enable_segmentation_rendering()
    renderer.update_scene(d, camera=cam)
    seg = renderer.render()
    # segmentation renders geometry ID in R channel. Robot is >0.
    mask = (seg[:, :, 0] > 0).astype(np.uint8) * 255
    os.makedirs("mj_mask_frames", exist_ok=True)
    Image.fromarray(mask).save(f"mj_mask_frames/frame_{step:04d}.png")
    renderer.disable_segmentation_rendering()
    
    evih_path = f"evih_headless_frames/frame_{step:04d}.png"
    if os.path.exists(evih_path):
        evih_img = Image.open(evih_path).convert("RGB")
        combined = Image.new("RGB", (1280, 480))
        combined.paste(evih_img, (0, 0))
        combined.paste(mj_pil, (640, 0))
        combined.save(f"side_by_side_frames/frame_{step:04d}.png")
    else:
        print(f"Missing {evih_path}")

print("Frames generated.")
