from pathlib import Path
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("MimicKitSkeletonReplay", "/mnt/d/AnimationTech-learning/EvihAnimation-mimickit-bridge/ai4animation/Standalone/MimicKitSkeletonReplay.py")
MODULE = importlib.util.module_from_spec(spec)
sys.modules["MimicKitSkeletonReplay"] = MODULE
spec.loader.exec_module(MODULE)

load_mesh_model = MODULE.load_mesh_model
load_replay_rows = MODULE.load_replay_rows
resolve_package_files = MODULE.resolve_package_files
posed_mesh_triangles = MODULE.posed_mesh_triangles
rasterize_triangles = MODULE.rasterize_triangles
_mask_stats = MODULE._mask_stats
write_png = MODULE.write_png

package_dir = Path('/root/Project/MimicKit/output/train/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/ue_export_mesh_reference')
mesh_asset = Path('/root/Project/MimicKit/output/img/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/assets/humanoid_sword_shield.glb')

files = resolve_package_files(package_dir)
rows = load_replay_rows(files["pose_dof_replay"])
import json
with open(package_dir / "scene_contract_v3.json") as f:
    contract = json.load(f)
with open(package_dir / "joint_order.json") as f:
    joint_order = json.load(f)
with open(package_dir / "mimickit_source_rig_asset_spec.json") as f:
    source_rig = json.load(f)
with open(package_dir / "mesh_binding_contract.json") as f:
    binding = json.load(f)
    
model = load_mesh_model(mesh_asset)

if "nodes" in binding and "renderable_bindings" not in binding:
    binding["renderable_bindings"] = []
    for n in binding.get("nodes", []):
        for i, node in enumerate(model.document.get("nodes", [])):
            if node.get("name") == n.get("node_name"):
                binding["renderable_bindings"].append({
                    "body_name": n.get("body_binding"),
                    "node_index": i
                })

frame_id = 5
import math
import copy

def axis_angle_matrix(axis, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return (
        (t * x * x + c, t * x * y + z * s, t * x * z - y * s, 0.0),
        (t * x * y - z * s, t * y * y + c, t * y * z + x * s, 0.0),
        (t * x * z + y * s, t * y * z - x * s, t * z * z + c, 0.0),
        (0.0, 0.0, 0.0, 1.0)
    )

def mat_mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )

def _zup_to_yup(r) -> None:
    pos = r.get("root_pos_m")
    if pos:
        r["root_pos_m"] = [float(pos[0]), float(pos[2]), -float(pos[1])]
    
    rot = r.get("root_rot_xyzw")
    if rot:
        rx = axis_angle_matrix([1.0, 0.0, 0.0], math.radians(-90.0))
        rx_inv = axis_angle_matrix([1.0, 0.0, 0.0], math.radians(90.0))
        
        qx, qy, qz, qw = float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])
        root_zup = (
            (1.0 - 2.0*qy*qy - 2.0*qz*qz, 2.0*qx*qy + 2.0*qz*qw, 2.0*qx*qz - 2.0*qy*qw, 0.0),
            (2.0*qx*qy - 2.0*qz*qw, 1.0 - 2.0*qx*qx - 2.0*qz*qz, 2.0*qy*qz + 2.0*qx*qw, 0.0),
            (2.0*qx*qz + 2.0*qy*qw, 2.0*qy*qz - 2.0*qx*qw, 1.0 - 2.0*qx*qx - 2.0*qy*qy, 0.0),
            (0.0, 0.0, 0.0, 1.0)
        )
        
        root_yup = mat_mul(rx, mat_mul(root_zup, rx_inv))
        m = root_yup
        tr = m[0][0] + m[1][1] + m[2][2]
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m[1][2] - m[2][1]) / S
            qy = (m[2][0] - m[0][2]) / S
            qz = (m[0][1] - m[1][0]) / S
        elif (m[0][0] > m[1][1]) and (m[0][0] > m[2][2]):
            S = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
            qw = (m[1][2] - m[2][1]) / S
            qx = 0.25 * S
            qy = (m[0][1] + m[1][0]) / S
            qz = (m[0][2] + m[2][0]) / S
        elif m[1][1] > m[2][2]:
            S = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
            qw = (m[2][0] - m[0][2]) / S
            qx = (m[0][1] + m[1][0]) / S
            qy = 0.25 * S
            qz = (m[1][2] + m[2][1]) / S
        else:
            S = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
            qw = (m[0][1] - m[1][0]) / S
            qx = (m[0][2] + m[2][0]) / S
            qy = (m[1][2] + m[2][1]) / S
            qz = 0.25 * S
        r["root_rot_xyzw"] = [qx, qy, qz, qw]

row = copy.deepcopy(rows[frame_id + 1]) # Because of the +1 offset
initial_row = copy.deepcopy(rows[0])
_zup_to_yup(row)
_zup_to_yup(initial_row)

triangles, _, _ = posed_mesh_triangles(model, row, joint_order, source_rig, binding, initial_row)




camera_for_frame = MODULE.camera_for_frame
root = tuple(float(value) for value in row["root_pos_m"])
eye, target, vfov_from_contract = camera_for_frame(contract, frame_id, root)

print("Camera Eye Zup:", eye)
print("Camera Target Zup:", target)
print("Camera VFOV from contract:", vfov_from_contract)
print("Converted Root Pos Yup:", row["root_pos_m"])

width, height = 960, 540

transform = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0)
)
test_fov = 45.0
rgb, silhouette = rasterize_triangles(triangles, transform, width=width, height=height, eye=eye, target=target, fov=test_fov)
write_png(Path('/tmp/test_squish.png'), width, height, rgb)
print("Saved to /tmp/test_squish.png")
