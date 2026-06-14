from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "ai4animation" / "Standalone" / "MimicKitSkeletonReplay.py"
SPEC = importlib.util.spec_from_file_location("evih_mimickit_replay_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load replay implementation: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

build_visual_metric_report = MODULE.build_visual_metric_report
build_true_mesh_replay = MODULE.build_true_mesh_replay
compare_scene_contract = MODULE.compare_scene_contract
default_thresholds = MODULE.default_thresholds
inspect_glb = MODULE.inspect_glb
render_true_mesh = MODULE.render_true_mesh
scene_contract_sha256 = MODULE.scene_contract_sha256
sha256_file = MODULE.sha256_file
validate_mesh_binding = MODULE.validate_mesh_binding
validate_mesh_reference_manifest = MODULE.validate_mesh_reference_manifest
write_png = MODULE.write_png


def write_glb(path: Path, document: dict, binary: bytes) -> None:
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    binary += b"\0" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    payload = struct.pack("<4sII", b"glTF", 2, total)
    payload += struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    payload += struct.pack("<II", len(binary), 0x004E4942) + binary
    path.write_bytes(payload)


def valid_rigid_glb(path: Path) -> None:
    positions = struct.pack("<9f", -0.4, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.8, 0.0)
    indices = struct.pack("<3H", 0, 1, 2)
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "pelvis", "mesh": 0, "children": [1, 2]},
            {"name": "sword", "mesh": 0, "translation": [0.7, 0.0, 0.0]},
            {"name": "shield", "mesh": 0, "translation": [-0.7, 0.0, 0.0]},
        ],
        "meshes": [{"name": "sword_shield_body", "primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(positions) + len(indices)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
    }
    write_glb(path, document, positions + indices)


def valid_skinned_glb(path: Path) -> None:
    positions = struct.pack("<9f", -0.4, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.8, 0.0)
    indices = struct.pack("<3H", 0, 1, 2) + b"\0\0"
    joints = bytes([0, 0, 0, 0] * 3)
    weights = struct.pack("<12f", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    binary = positions + indices + joints + weights
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "pelvis", "children": [1, 2, 3]},
            {"name": "sword"},
            {"name": "shield"},
            {"name": "avatar_mesh", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1, 2]}],
        "meshes": [
            {
                "name": "sword_shield_body",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3},
                        "indices": 1,
                    }
                ],
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": 6},
            {"buffer": 0, "byteOffset": len(positions) + len(indices), "byteLength": len(joints)},
            {"buffer": 0, "byteOffset": len(positions) + len(indices) + len(joints), "byteLength": len(weights)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5121, "count": 3, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"},
        ],
    }
    write_glb(path, document, binary)


def mask_pixels(width: int, height: int, x0: int) -> bytes:
    pixels = bytearray((0, 0, 0) * (width * height))
    for y in range(2, 6):
        for x in range(x0, x0 + 4):
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = b"\xff\xff\xff"
    return bytes(pixels)


class Plan63V2Tests(unittest.TestCase):
    def test_empty_glb_is_rejected_and_valid_rigid_glb_binds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty.glb"
            write_glb(empty, {"asset": {"version": "2.0"}, "buffers": []}, b"")
            self.assertFalse(inspect_glb(empty)["ok"])

            asset = root / "valid.glb"
            valid_rigid_glb(asset)
            structure = inspect_glb(asset)
            self.assertTrue(structure["ok"], structure)
            joint_order = {"body_order": ["pelvis", "sword", "shield"], "dof_size": 1, "joints": []}
            binding = validate_mesh_binding(asset, joint_order, structure)
            self.assertTrue(binding["mesh_binding_pass"], binding)
            self.assertFalse(binding["xml_geom_fallback_used"])

            skinned_asset = root / "skinned.glb"
            valid_skinned_glb(skinned_asset)
            skinned_structure = inspect_glb(skinned_asset)
            skinned_binding = validate_mesh_binding(skinned_asset, joint_order, skinned_structure)
            self.assertTrue(skinned_binding["mesh_binding_pass"], skinned_binding)
            self.assertEqual(skinned_binding["binding_mode"], "skinned")

    def test_scene_contract_hash_and_camera_coverage_are_required(self) -> None:
        contract = {
            "schema_version": 2,
            "resolution": {"width": 64, "height": 64},
            "timing": {"frames": 10, "frame_stride": 5, "fps": 12, "expected_frame_ids": [0, 5]},
            "camera_samples": [
                {"frame": 0, "eye_m": [0, -5, 3], "target_m": [0, 0, 1], "fov_degrees": 45},
                {"frame": 5, "eye_m": [0, -5, 3], "target_m": [0, 0, 1], "fov_degrees": 45},
            ],
        }
        contract["scene_contract_sha256"] = scene_contract_sha256(contract)
        report = compare_scene_contract(contract, width=64, height=64, stride=5, fps=12, frame_ids=[0, 5])
        self.assertTrue(report["scene_contract_compare_pass"], report)
        contract["camera_samples"].pop()
        report = compare_scene_contract(contract, width=64, height=64, stride=5, fps=12, frame_ids=[0, 5])
        self.assertFalse(report["scene_contract_compare_pass"])

    def test_exact_silhouettes_pass_requested_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source: dict[int, Path] = {}
            evih: dict[int, Path] = {}
            for frame, x0 in ((0, 2), (5, 5)):
                source_path = root / "source" / f"frame_{frame:06d}.png"
                evih_path = root / "evih" / f"frame_{frame:06d}.png"
                write_png(source_path, 16, 10, mask_pixels(16, 10, x0))
                write_png(evih_path, 16, 10, mask_pixels(16, 10, x0))
                source[frame] = source_path
                evih[frame] = evih_path
            report = build_visual_metric_report(source, evih, thresholds=default_thresholds())
            self.assertTrue(report["visual_metric_pass"], report)
            self.assertTrue(report["metrics"]["motion_visible"])

    def test_failed_mimickit_manifest_cannot_enter_true_mesh_replay(self) -> None:
        manifest = {
            "mesh_reference_pass": False,
            "render_summary": {
                "visual_kind": "mesh",
                "mesh_detected": True,
                "image_count": 2,
                "expected_image_count": 2,
                "frame_ids": [0, 5],
                "expected_frame_ids": [0, 5],
                "mp4_ok": True,
            },
        }
        report = validate_mesh_reference_manifest(manifest)
        self.assertFalse(report["ok"])
        self.assertEqual(report["blocker"], "mimickit_mesh_reference_not_passing")

    def test_synthetic_end_to_end_true_mesh_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            (package / "visual_replay").mkdir(parents=True)
            rows = [
                {"frame": 0, "root_pos_m": [0.0, 0.0, 0.0], "root_rot_xyzw": [0.0, 0.0, 0.0, 1.0], "dof_pos": [0.0]},
                {"frame": 5, "root_pos_m": [0.5, 0.0, 0.0], "root_rot_xyzw": [0.0, 0.0, 0.0, 1.0], "dof_pos": [0.1]},
            ]
            joint_order = {
                "body_order": ["pelvis", "sword", "shield"],
                "dof_size": 1,
                "joints": [
                    {
                        "name": "sword",
                        "body_name": "sword",
                        "parent_body_name": "pelvis",
                        "joint_type": "hinge",
                        "dof_index": 0,
                        "dof_dim": 1,
                        "dof_slice": [0, 1],
                    }
                ],
            }
            (package / "visual_replay" / "pose_dof_replay.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            for path, payload in (
                (package / "visual_replay" / "pose_dof_meta.json", {"schema_version": 1}),
                (package / "joint_order.json", joint_order),
                (package / "mimickit_source_rig_asset_spec.json", {"schema_version": 1}),
                (package / "visual_alignment_contract.json", {"schema_version": 1}),
            ):
                path.write_text(json.dumps(payload), encoding="utf-8")
            asset = root / "valid.glb"
            valid_rigid_glb(asset)
            contract = {
                "schema_version": 2,
                "resolution": {"width": 64, "height": 64},
                "timing": {"frames": 10, "frame_stride": 5, "fps": 12, "expected_frame_ids": [0, 5]},
                "camera_samples": [
                    {"frame": 0, "eye_m": [0, -5, 3], "target_m": [0, 0, 0], "fov_degrees": 45},
                    {"frame": 5, "eye_m": [0, -5, 3], "target_m": [0, 0, 0], "fov_degrees": 45},
                ],
            }
            contract["scene_contract_sha256"] = scene_contract_sha256(contract)
            source_render = root / "source_render"
            render_true_mesh(rows, [0, 5], asset, joint_order, contract, source_render, width=64, height=64)
            manifest = {
                "mesh_reference_pass": True,
                "scene_contract_v2": contract,
                "asset_export": {"output_sha256": sha256_file(asset)},
                "render_summary": {
                    "render_dir": str(source_render),
                    "visual_kind": "mesh",
                    "mesh_detected": True,
                    "image_count": 2,
                    "expected_image_count": 2,
                    "frame_ids": [0, 5],
                    "expected_frame_ids": [0, 5],
                    "mp4_ok": True,
                    "source_was_ppm_only": False,
                },
            }
            manifest_path = root / "mesh_reference_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            visual_review = root / "visual_review.json"
            visual_review.write_text(
                json.dumps(
                    {
                        "visual_review_pass": True,
                        "checks": {
                            "character": True,
                            "sword": True,
                            "shield": True,
                            "pose": True,
                            "camera": True,
                            "ground": True,
                            "lighting": True,
                            "frame_pairing": True,
                            "no_obvious_penetration_or_drift": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = build_true_mesh_replay(
                package_dir=package,
                mesh_asset=asset,
                mesh_reference_manifest_path=manifest_path,
                out_dir=root / "evih",
                visual_review_path=visual_review,
                thresholds=default_thresholds(),
            )
            self.assertTrue(report["evih_mesh_replay_pass"], report)
            self.assertTrue(report["mesh_binding_pass"])
            self.assertTrue(report["scene_contract_compare_pass"])
            self.assertTrue(report["visual_metric_pass"])
            self.assertTrue(report["visual_review_pass"])
            self.assertTrue((root / "evih" / "mimickit_mesh_vs_evih_mesh_sheet.png").is_file())


if __name__ == "__main__":
    unittest.main()
