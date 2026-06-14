import json
import math
from pathlib import Path
import sys

import importlib.util

REPO_ROOT = Path(__file__).resolve().parent
MODULE_PATH = REPO_ROOT / "ai4animation" / "Standalone" / "MimicKitSkeletonReplay.py"
SPEC = importlib.util.spec_from_file_location("evih_mimickit_replay", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

_mask_stats = MODULE._mask_stats
load_replay_rows = MODULE.load_replay_rows
read_json = MODULE.read_json
resolve_package_files = MODULE.resolve_package_files
validate_mesh_binding = MODULE.validate_mesh_binding
inspect_glb = MODULE.inspect_glb
pose_node_world_matrices = MODULE.pose_node_world_matrices
posed_mesh_triangles = MODULE.posed_mesh_triangles
rasterize_triangles = MODULE.rasterize_triangles
camera_for_frame = MODULE.camera_for_frame
load_mesh_model = MODULE.load_mesh_model

manifest_path = Path(r'\\wsl.localhost\Ubuntu-20.04\root\Project\MimicKit\output\train\tmp_white_knight_mesh_reference_20260607_bridge_full_v2\mesh_reference_manifest.json')
package_dir = Path(r'\\wsl.localhost\Ubuntu-20.04\root\Project\MimicKit\output\train\tmp_white_knight_mesh_reference_20260607_bridge_full_v2\ue_export_mesh_reference')
mesh_asset = Path(r'\\wsl.localhost\Ubuntu-20.04\root\Project\MimicKit\output\img\tmp_white_knight_mesh_reference_20260607_bridge_full_v2\assets\humanoid_sword_shield.glb')
source_silhouettes_dir = Path(r'\\wsl.localhost\Ubuntu-20.04\root\Project\MimicKit\output\img\tmp_white_knight_mesh_reference_20260607_bridge_full_v2\runs\view_motion_humanoid_sword_shield_args\RL_Avatar_Atk_2xCombo01_Motion\render\silhouettes')

files = resolve_package_files(package_dir)
rows = load_replay_rows(files["pose_dof_replay"])
joint_order = read_json(files["joint_order"])
source_rig = read_json(files["source_rig_asset_spec"])
structure = inspect_glb(mesh_asset)
binding = validate_mesh_binding(mesh_asset, joint_order, structure, mesh_binding_contract=read_json(package_dir / "mesh_binding_contract.json"))
contract = read_json(package_dir / "scene_contract_v3.json")

frame_id = 5
row = rows[frame_id]
initial_row = rows[0]

model = load_mesh_model(mesh_asset)

source_width, source_height, source_mask, source_centroid, source_bbox = _mask_stats(source_silhouettes_dir / f"frame_{frame_id:06d}.png")

def test_vfov(test_vfov):
    eye, target, _ = camera_for_frame(contract, frame_id, row.get("root_pos_m", [0,0,0]))
    triangles, _, _ = posed_mesh_triangles(model, row, joint_order, source_rig, binding, initial_row)
    width = 960
    height = 540
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    rgb, silhouette = rasterize_triangles(triangles, identity, width=width, height=height, eye=eye, target=target, fov=test_vfov)
    
    evih_mask = set()
    for index in range(width * height):
        offset = index * 3
        if max(silhouette[offset : offset + 3]) >= 128:
            evih_mask.add(index)
            
    union = len(source_mask | evih_mask)
    intersection = len(source_mask & evih_mask)
    iou = intersection / union if union else 0.0
    return iou

best_iou = 0
best_vfov = 0
for vfov_int in range(200, 800, 5):
    vfov = vfov_int / 10.0
    iou = test_vfov(vfov)
    print(f"vfov: {vfov:.1f}, iou: {iou:.3f}")
    if iou > best_iou:
        best_iou = iou
        best_vfov = vfov

print(f"Best vfov: {best_vfov} with iou {best_iou}")
