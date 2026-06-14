import sys
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('MimicKitSkeletonReplay', '/mnt/d/AnimationTech-learning/EvihAnimation-mimickit-bridge/ai4animation/Standalone/MimicKitSkeletonReplay.py')
MODULE = importlib.util.module_from_spec(spec)
sys.modules['MimicKitSkeletonReplay'] = MODULE
spec.loader.exec_module(MODULE)

mesh_asset = Path(r'\\wsl.localhost\Ubuntu-20.04\root\Project\MimicKit\output\img\tmp_white_knight_mesh_reference_20260607_bridge_full_v2\assets\humanoid_sword_shield.glb')
model = MODULE.load_mesh_model(mesh_asset)
for i, n in enumerate(model.document.get('nodes', [])):
    if 'name' in n:
        print(f"{n.get('name')}: pos={n.get('translation', [0,0,0])}")
