import os
import sys
import json
import numpy as np
from pathlib import Path
from ai4animation import (
    AI4Animation,
    Actor,
    Vector3,
    Time
)

# This script is designed to run in Windows and load Sidecars from WSL
# E.g. \\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks_lite_plan_skill_pack\output\motionbricks_lite\exports\demo_run_01

class Program:
    def __init__(self, export_dir=None):
        self.export_dir = export_dir
        self.qpos_sequence = None
        self.contract = None
        self.manifest = None
        self.Actor = None
        self.total_frames = 0
        self.fps = 30.0

    def load_sidecars(self):
        if not self.export_dir or not os.path.exists(self.export_dir):
            print("No valid export directory provided or found. Running in idle/dummy mode.")
            return

        # 1. Load manifest
        manifest_path = os.path.join(self.export_dir, "motionbricks_replay_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                self.manifest = json.load(f)
                self.total_frames = self.manifest.get("num_frames", 0)
                self.fps = self.manifest.get("fps", 30.0)

        # 2. Load contract
        contract_path = os.path.join(self.export_dir, "coordinate_contract.json")
        if os.path.exists(contract_path):
            with open(contract_path, "r") as f:
                self.contract = json.load(f)

        # 3. Load qpos (or joint_transforms)
        qpos_path = os.path.join(self.export_dir, "qpos.npy")
        if os.path.exists(qpos_path):
            self.qpos_sequence = np.load(qpos_path)
            print(f"Loaded qpos sequence shape: {self.qpos_sequence.shape}")

    def Start(self):
        self.load_sidecars()

        # Initialize Actor (Skeleton / Mesh)
        # Using a dummy model for now until we convert G1 mesh to GLB/FBX
        entity = AI4Animation.Scene.AddEntity("G1_Actor")
        
        # In the future, we load the specific G1 model here
        # model_path = os.path.join(self.export_dir, "g1_model.glb")
        # self.Actor = entity.AddComponent(Actor, model_path, None)

    def Standalone(self):
        pass

    def Update(self):
        if self.qpos_sequence is None or self.total_frames == 0:
            return

        # Frame calculation
        current_frame = int(Time.TotalTime * self.fps) % self.total_frames
        
        # TODO: Decode qpos to joint transforms using contract
        # current_qpos = self.qpos_sequence[current_frame]
        
        # Apply transforms to Actor
        # if self.Actor:
        #     self.Actor.SetTransforms(...)
        #     self.Actor.SyncToScene()

if __name__ == "__main__":
    # Point this to the WSL export path
    # Make sure WSL is running so UNC path resolves
    target_path = r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks_lite_plan_skill_pack\output\motionbricks_lite\exports\demo_run_01"
    
    AI4Animation(Program(export_dir=target_path))
