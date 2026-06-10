import os
import sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent.parent))

# Mock raylib before importing Program
import types
sys.modules['raylib'] = types.ModuleType('raylib')
import raylib as rl
rl.KEY_LEFT_SHIFT = 0
rl.KEY_Q = 1
rl.KEY_E = 2
rl.MOUSE_BUTTON_RIGHT = 3

rl.IsKeyDown = lambda key: False
rl.IsKeyPressed = lambda key: False
rl.IsMouseButtonDown = lambda btn: False

from ai4animation import AI4Animation, Scene
from ai4animation.Standalone.OfflineMeshRenderer import OfflineMeshRenderer
from ai4animation.Import.GLBImporter import GLB

class MockCamera:
    def SetTarget(self, entity):
        pass

class MockIO:
    def GamepadAvailable(self):
        return False
        
    def GetWASDQE(self):
        # Always move forward
        return [0, 0, -1] # X, Y, Z
        
    def GetMousePositionOnScreen(self):
        return [0, 0]
        
    def LogErrorIfGamepadNotAvailable(self):
        pass

class GenericMock:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return GenericMock()
    def __getattr__(self, name):
        return GenericMock()
    def __setattr__(self, name, value):
        self.__dict__[name] = value

class MockGUI:
    Handle = GenericMock
    Canvas = GenericMock
    Button = GenericMock
    Dropdown = GenericMock
    HorizontalBar = GenericMock
    HorizontalPivot = GenericMock
    BarPlot = GenericMock

class MockStandalone:
    def __init__(self):
        self.Camera = MockCamera()
        self.IO = MockIO()
        
    def CreateSkinnedMesh(self, actor, model):
        return None

def main():
    AI4Animation.GUI = MockGUI
    AI4Animation.Standalone = MockStandalone()
    AI4Animation.Scene = Scene()
    
    # Import program after mocking
    from Program import Program
    
    prog = Program()
    prog.Start()
    
    model_path = os.path.join(str(SCRIPT_DIR.parent.parent / "_ASSETS_/Geno"), "Model.glb")
    glb = GLB.Create(model_path)
    
    global_matrices_seq = []
    
    print("Simulating Locomotion Biped...")
    num_frames = 300
    for frame in range(num_frames):
        # Change direction halfway
        if frame > 150:
            prog.GuidanceStyleIndex = 1
            prog._set_guidance(1)
            
        prog.Update()
        
        # Record the full global matrices of all glb nodes
        actor = prog.Actor
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
    out_path = str(SCRIPT_DIR / "locomotion_biped_mesh.mp4")
    renderer.render_animation(global_matrices_seq, out_path, fps=30)

if __name__ == "__main__":
    main()
