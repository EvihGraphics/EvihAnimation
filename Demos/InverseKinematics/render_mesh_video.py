import os
import sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from ai4animation import AI4Animation, Actor, FABRIK, Scene
from ai4animation.Standalone.OfflineMeshRenderer import OfflineMeshRenderer
from ai4animation.Import.GLBImporter import GLB

ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_/Geno")
sys.path.append(ASSETS_PATH)
import Definitions

def main():
    AI4Animation.Standalone = None
    AI4Animation.Scene = Scene()
    
    actor_entity = AI4Animation.Scene.AddEntity("Actor")
    model_path = os.path.join(ASSETS_PATH, "Model.glb")
    
    actor = actor_entity.AddComponent(
        Actor, model_path, None, False
    )

    glb = GLB.Create(model_path)

    ik = FABRIK(
        actor.GetBone(Definitions.LeftShoulderName),
        actor.GetBone(Definitions.LeftWristName),
    )

    initial_wrist_pos = actor.GetBone(Definitions.LeftWristName).GetPosition()
    pose = actor.GetTransforms().copy()
    
    global_matrices_seq = []
    
    print("Simulating Inverse Kinematics...")
    num_frames = 150
    for frame in range(num_frames):
        # Create a circular motion for the target
        t = frame / 30.0 * np.pi * 2.0
        target_pos = initial_wrist_pos + np.array([np.sin(t) * 0.15, np.cos(t) * 0.15, 0.0])
        target_rot = np.eye(3)
        
        # Reset to base pose
        actor.SetTransforms(pose)
        
        # Solve IK
        ik.Solve(
            target_pos,
            target_rot,
            max_iterations=10,
            threshold=0.001,
        )
        
        # We also need to compute FK for all other joints to update them.
        # But wait, actor.SyncToScene updates the entities. 
        # Actually Actor doesn't compute FK for the whole skeleton automatically in headless mode unless requested?
        # Actually, in Actor.py, SetTransforms sets the global matrices directly. IK solves and sets local rotations probably?
        # Wait, FABRIK updates global transforms directly!
        actor.SyncToScene(ik.Bones)
        
        # Record the full global matrices of all glb nodes
        actor_transforms = actor.GetTransforms()
        glb_global = np.zeros((len(glb._nodes), 4, 4), dtype=np.float32)
        for i, node in enumerate(glb._nodes):
            if node.Parent is None:
                glb_global[i] = node.LocalMatrix
            else:
                glb_global[i] = glb_global[node.Parent] @ node.LocalMatrix
            
        actor_bone_names = actor.GetBoneNames()
        for b_idx, b_name in enumerate(actor_bone_names):
            if b_name in glb._nodeNames:
                glb_idx = glb._nodeNames.index(b_name)
                glb_global[glb_idx] = actor_transforms[b_idx]
                
        global_matrices_seq.append(glb_global)

    print("Rendering Sequence...")
    renderer = OfflineMeshRenderer(glb, width=1280, height=720)
    out_path = str(SCRIPT_DIR / "ik_mesh.mp4")
    renderer.render_animation(global_matrices_seq, out_path, fps=30)
        
if __name__ == "__main__":
    main()
