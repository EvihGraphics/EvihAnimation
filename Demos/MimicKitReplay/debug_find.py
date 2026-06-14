import importlib.util, sys
from pathlib import Path
MODULE_PATH = Path('/mnt/d/AnimationTech-learning/EvihAnimation-mimickit-bridge/ai4animation/Standalone/MimicKitSkeletonReplay.py')
SPEC = importlib.util.spec_from_file_location('evih_mimickit_replay', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

import json
manifest_path = Path('/mnt/d/MimicKitNative/workspace/MimicKit/output/train/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/mesh_reference_manifest.json')
manifest = json.loads(manifest_path.read_text())
bases = [manifest_path.parent]
render_values = MODULE._recursive_values(manifest, {'render_dir', 'frames_dir', 'rgb_frames_dir'})
print('render_values:', render_values)
render_dir = MODULE._first_existing_path(render_values, bases)
print('render_dir:', render_dir)

rgb, sil = MODULE.locate_reference_frames(manifest, manifest_path, [0, 5])
print('rgb:', rgb)
print('sil:', sil)
