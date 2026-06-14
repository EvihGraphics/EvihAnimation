from __future__ import annotations

import importlib.util
import json
import math
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
build_dynamic_sequence_report = MODULE.build_dynamic_sequence_report
build_rgb_metric_report = MODULE.build_rgb_metric_report
build_dynamic_comparison_mp4 = MODULE.build_dynamic_comparison_mp4
build_true_mesh_replay = MODULE.build_true_mesh_replay
compare_scene_contract = MODULE.compare_scene_contract
compare_consumed_hashes = MODULE.compare_consumed_hashes
create_mp4 = MODULE.create_mp4
default_thresholds = MODULE.default_thresholds
ensure_visual_review_template = MODULE.ensure_visual_review_template
apply_ground_shadows = MODULE.apply_ground_shadows
inspect_glb = MODULE.inspect_glb
render_true_mesh = MODULE.render_true_mesh
render_ground_scene = MODULE.render_ground_scene
quat_matrix = MODULE.quat_matrix
quat_rotate_vec = MODULE.quat_rotate_vec
transform_point = MODULE.transform_point
scene_contract_sha256 = MODULE.scene_contract_sha256
sha256_file = MODULE.sha256_file
validate_mesh_binding = MODULE.validate_mesh_binding
validate_mesh_reference_manifest = MODULE.validate_mesh_reference_manifest
validate_framework_renderer_provenance = MODULE.validate_framework_renderer_provenance
validate_visual_review = MODULE.validate_visual_review
write_png = MODULE.write_png
write_comparison_sheet = MODULE.write_comparison_sheet
write_comparison_markdown = MODULE.write_comparison_markdown


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


def binding_contract(names: list[str]) -> dict:
    return {
        "schema_version": 1,
        "nodes": [{"node_name": name, "body_binding": name} for name in names],
    }


def strict_scene_contract(width: int = 64, height: int = 64, frame_ids: tuple[int, ...] = (0, 5)) -> dict:
    samples = [
        {
            "frame": frame,
            "eye": [0.0, -5.0, 3.0],
            "target": [0.0, 0.0, 0.5],
            "fov_degrees": 60.0,
            "fov_axis": "horizontal",
            "projection": "perspective",
            "near": 0.1,
            "far": 1000.0,
            "visual_link_sync_ok": True,
            "visual_link_sync_count": 1,
            "visual_link_sync_max_pos_error_m": 0.0,
            "visual_link_sync_max_rot_error_rad": 0.0,
            "renderer_version": "test",
            "capture_settle_updates": 8,
        }
        for frame in frame_ids
    ]
    contract = {
        "schema_version": 3,
        "resolution": {"width": width, "height": height},
        "timing": {"frames": frame_ids[-1] + 5, "frame_stride": 5, "fps": 12, "expected_frame_ids": list(frame_ids)},
        "source": {"capture_index_sha256": "a" * 64},
        "coordinate_system": {"basis": "mimickit_training_xyz_z_up"},
        "camera": {"mode": "track", "projection": "perspective"},
        "ground": {
            "kind": "flat_grid_plane",
            "color_rgb": [0.017, 0.0153, 0.01275],
            "albedo_add": 10.0,
            "grid_spacing_m": 0.5,
            "major_grid_spacing_m": 5.0,
        },
        "lights": {
            "distant": {
                "intensity": 2000.0,
                "color_rgb": [0.8, 0.8, 0.8],
                "rotation_euler_xyz_rad": [0.7, 0.0, 0.6],
                "direction_world": [-0.36375266832671915, 0.5316958010320105, -0.7648421872844884],
                "casts_shadows": True,
            },
            "dome": {"intensity": 800.0, "color_rgb": [0.7, 0.7, 0.7]},
        },
        "renderer": "isaaclab",
        "color_space": "srgb",
        "debug_overlays": False,
        "capture": {"settle_updates_per_attempt": [8], "requires_visual_link_sync": True},
        "camera_samples": samples,
    }
    contract["scene_contract_sha256"] = scene_contract_sha256(contract)
    return contract


class Plan63V2Tests(unittest.TestCase):
    def test_framework_grid_shader_uses_contract_spacing_in_world_coordinates(self) -> None:
        shader = (
            REPO_ROOT / "ai4animation" / "Standalone" / "resources" / "shaders" / "grid.fs"
        ).read_text(encoding="utf-8")
        self.assertIn("uniform float gridSpacing;", shader)
        self.assertIn("uniform float majorGridSpacing;", shader)
        self.assertIn("vec2 worldXZ = fragPosition.xz;", shader)
        self.assertNotIn("20.0 * 10.0 * fragTexCoord", shader)

    def test_framework_render_pipeline_exposes_contract_sky_strength(self) -> None:
        source = (
            REPO_ROOT / "ai4animation" / "Standalone" / "RenderPipeline.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self.SkyStrength = 0.15", source)
        self.assertIn("skyStrengthPtr[0] = self.SkyStrength", source)
        self.assertNotIn("skyStrengthPtr[0] = 0.15", source)

    def test_ground_scene_uses_contract_color_and_grid(self) -> None:
        contract = strict_scene_contract()
        mask, rgb = render_ground_scene(64, 64, (0.0, -5.0, 3.0), (0.0, 0.0, 0.5), 35.0, contract)
        self.assertGreater(sum(mask), 0)
        self.assertGreater(len({rgb[index : index + 3] for index in range(0, len(rgb), 3)}), 1)
        shadowed = apply_ground_shadows(
            rgb,
            mask,
            [MODULE.MeshTriangle(points=((-0.5, 0.0, 1.0), (0.5, 0.0, 1.0), (0.0, 0.5, 1.0)), color=(255, 255, 255))],
            width=64,
            height=64,
            eye=(0.0, -5.0, 3.0),
            target=(0.0, 0.0, 0.5),
            fov=35.0,
            contract=contract,
        )
        self.assertNotEqual(rgb, shadowed)
        changed_strengths = {
            sum(rgb[offset + channel] - shadowed[offset + channel] for channel in range(3))
            for offset in range(0, len(rgb), 3)
            if rgb[offset : offset + 3] != shadowed[offset : offset + 3]
        }
        self.assertGreater(len(changed_strengths), 1, "ground shadow should have a soft opacity gradient")

    def test_rgb_metric_report_is_report_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            evih = root / "evih.png"
            pixels = bytes((100, 110, 120) * 16)
            write_png(source, 4, 4, pixels)
            write_png(evih, 4, 4, pixels)
            report = build_rgb_metric_report({0: source}, {0: evih})
            self.assertTrue(report["report_only"])
            self.assertTrue(report["complete"])
            self.assertEqual(report["metrics"]["mean_absolute_error_normalized"], 0.0)

    def test_comparison_markdown_preserves_manual_review_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "frame_000000.png"
            evih = root / "evih_frame_000000.png"
            sheet = root / "sheet.png"
            pixels = bytes((100, 110, 120) * 16)
            for path in (source, evih, sheet):
                write_png(path, 4, 4, pixels)
            report = write_comparison_markdown(
                root / "comparison_sheet.md",
                pairs=[(source, evih)],
                comparison_sheet=sheet,
                visual_metric_report={"metrics": {}},
                ground_metric_report={"metrics": {}},
                rgb_metric_report={"metrics": {}},
                visual_review={"visual_review_pass": False, "review": {"checks": {}}},
                dynamic_sequence_report={"dynamic_sequence_pass": True, "required_frame_count": 60},
            )
            self.assertTrue(report["ok"])
            text = (root / "comparison_sheet.md").read_text(encoding="utf-8")
            self.assertIn("Final visual review pass: **False**", text)
            self.assertIn("Dynamic sequence pass | True", text)
            self.assertNotIn("- [x] character", text)

    def test_comparison_sheet_samples_full_dynamic_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = []
            pixels = bytes((100, 110, 120) * 16)
            for frame in range(20):
                source = root / "source" / f"frame_{frame:06d}.png"
                evih = root / "evih" / f"frame_{frame:06d}.png"
                write_png(source, 4, 4, pixels)
                write_png(evih, 4, 4, pixels)
                pairs.append((source, evih))
            report = write_comparison_sheet(pairs, root / "sheet.png", sample_count=4, thumb_width=4, thumb_height=4)
            self.assertEqual(report["selected_frame_ids"][0], 0)
            self.assertEqual(report["selected_frame_ids"][-1], 19)
            self.assertEqual(len(report["selected_frame_ids"]), 4)

    def test_dynamic_comparison_mp4_contains_full_paired_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = []
            for index, frame in enumerate(MODULE.V3_REQUIRED_FRAME_IDS):
                source = root / "source" / f"frame_{frame:06d}.png"
                evih = root / "evih" / f"frame_{frame:06d}.png"
                write_png(source, 4, 4, bytes((index % 256, 10, 20) * 16))
                write_png(evih, 4, 4, bytes((30, index % 256, 40) * 16))
                pairs.append((source, evih))
            report = build_dynamic_comparison_mp4(pairs, root / "comparison", fps=12)
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["frame_count"], 60)
            self.assertEqual(report["comparison_png_count"], 60)

    def test_static_or_missing_framework_media_fails_dynamic_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_ids = list(MODULE.V3_REQUIRED_FRAME_IDS)
            source_rgb = {}
            source_silhouettes = {}
            evih_rgb = {}
            evih_silhouettes = {}
            for frame_id in frame_ids:
                for name, mapping in (
                    ("source_rgb", source_rgb),
                    ("source_silhouette", source_silhouettes),
                    ("evih_rgb", evih_rgb),
                    ("evih_silhouette", evih_silhouettes),
                ):
                    path = root / name / f"frame_{frame_id:06d}.png"
                    write_png(path, 4, 4, bytes((40, 50, 60) * 16))
                    mapping[frame_id] = path
            media = create_mp4([evih_rgb[frame_id] for frame_id in frame_ids], root / "static.mp4", 12)
            static = build_dynamic_sequence_report(
                frame_ids=frame_ids,
                source_rgb=source_rgb,
                source_silhouettes=source_silhouettes,
                evih_rgb=evih_rgb,
                evih_silhouettes=evih_silhouettes,
                media=media,
            )
            self.assertFalse(static["dynamic_sequence_pass"])
            self.assertIn("unique", static["blocker"])
            missing = build_dynamic_sequence_report(
                frame_ids=frame_ids,
                source_rgb=source_rgb,
                source_silhouettes=source_silhouettes,
                evih_rgb=evih_rgb,
                evih_silhouettes=evih_silhouettes,
                media={},
            )
            self.assertFalse(missing["dynamic_sequence_pass"])
            self.assertEqual(missing["blocker"], "dynamic_sequence_failed:mp4_frame_count_exact")

    def test_visual_review_requires_named_reviewer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory) / "visual_review.json"
            evidence = {"schema_version": 1, "comparison_sheet_sha256": "a" * 64}
            review.write_text(
                json.dumps(
                    {
                        "visual_review_pass": True,
                        "reviewer": "",
                        "reviewed_at_utc": "2026-06-12T00:00:00Z",
                        "checks": {name: True for name in MODULE.REQUIRED_VISUAL_REVIEW_CHECKS},
                        "evidence": evidence,
                    }
                ),
                encoding="utf-8",
            )
            report = MODULE.validate_visual_review(review, evidence)
            self.assertFalse(report["visual_review_pass"])
            self.assertFalse(report["reviewer_present"])
            payload = json.loads(review.read_text(encoding="utf-8"))
            payload["reviewer"] = "test-reviewer"
            payload["reviewed_at_utc"] = ""
            review.write_text(json.dumps(payload), encoding="utf-8")
            report = MODULE.validate_visual_review(review, evidence)
            self.assertFalse(report["visual_review_pass"])
            self.assertFalse(report["reviewed_at_present"])

    def test_visual_review_is_bound_to_current_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory) / "visual_review.json"
            evidence = {"schema_version": 1, "comparison_sheet_sha256": "a" * 64}
            self.assertTrue(ensure_visual_review_template(review, evidence))
            payload = json.loads(review.read_text(encoding="utf-8"))
            payload["visual_review_pass"] = True
            payload["reviewer"] = "test-reviewer"
            payload["reviewed_at_utc"] = "2026-06-12T00:00:00Z"
            payload["checks"] = {name: True for name in MODULE.REQUIRED_VISUAL_REVIEW_CHECKS}
            review.write_text(json.dumps(payload), encoding="utf-8")
            current = validate_visual_review(review, evidence)
            self.assertTrue(current["visual_review_pass"], current)
            stale = validate_visual_review(review, {"schema_version": 1, "comparison_sheet_sha256": "b" * 64})
            self.assertFalse(stale["visual_review_pass"])
            self.assertEqual(stale["blocker"], "visual_review_evidence_mismatch")

    def test_stale_signed_review_is_archived_and_replaced_with_current_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory) / "visual_review.json"
            old_evidence = {"schema_version": 1, "comparison_sheet_sha256": "a" * 64}
            new_evidence = {"schema_version": 2, "comparison_sheet_sha256": "b" * 64}
            ensure_visual_review_template(review, old_evidence)
            payload = json.loads(review.read_text(encoding="utf-8"))
            payload["visual_review_pass"] = True
            payload["reviewer"] = "test-reviewer"
            payload["reviewed_at_utc"] = "2026-06-12T00:00:00Z"
            payload["checks"] = {name: True for name in MODULE.REQUIRED_VISUAL_REVIEW_CHECKS}
            review.write_text(json.dumps(payload), encoding="utf-8")

            self.assertTrue(ensure_visual_review_template(review, new_evidence))
            archives = list(review.parent.glob("visual_review.stale.*.json"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(json.loads(archives[0].read_text(encoding="utf-8"))["reviewer"], "test-reviewer")
            current = json.loads(review.read_text(encoding="utf-8"))
            self.assertEqual(current["evidence"], new_evidence)
            self.assertFalse(current["visual_review_pass"])
            self.assertEqual(current["reviewer"], "")
            self.assertFalse(any(current["checks"].values()))

    def test_framework_provenance_requires_exact_api_and_bound_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = {
                **MODULE.FRAMEWORK_PROVENANCE_EXACT,
                "capture_passes": ["blank", "character_only", "ground_only", "full_scene"],
                "python_executable": sys.executable,
            }
            expected_paths = {}
            for name in MODULE.FRAMEWORK_PROVENANCE_ARTIFACTS:
                path = root / name
                path.write_text(name, encoding="utf-8")
                provenance[name] = str(path)
                provenance[f"{name}_sha256"] = sha256_file(path)
                expected_paths[name] = path
            report = validate_framework_renderer_provenance(provenance, expected_paths)
            self.assertTrue(report["framework_renderer_provenance_valid"], report)

            provenance["actor_component"] = "fake.Actor"
            self.assertFalse(
                validate_framework_renderer_provenance(provenance, expected_paths)[
                    "framework_renderer_provenance_valid"
                ]
            )
            provenance["actor_component"] = MODULE.FRAMEWORK_PROVENANCE_EXACT["actor_component"]
            Path(provenance["body_world_replay"]).write_text("drift", encoding="utf-8")
            self.assertFalse(
                validate_framework_renderer_provenance(provenance, expected_paths)[
                    "framework_renderer_provenance_valid"
                ]
            )

    def test_scene_consumed_hash_is_computed_not_copied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "mesh.glb"
            asset.write_bytes(b"mesh")
            contract = strict_scene_contract()
            contract["scene_contract_sha256"] = "f" * 64
            report = compare_consumed_hashes({}, {"hashes": {}}, asset, contract)
            scene = report["hashes"]["scene_contract"]
            self.assertEqual(scene["source_sha256"], "f" * 64)
            self.assertEqual(scene["evih_consumed_sha256"], scene_contract_sha256(contract))
            self.assertNotEqual(scene["evih_consumed_sha256"], scene["source_sha256"])

    def test_row_vector_quaternion_matrix_matches_fk_rotation(self) -> None:
        half = math.pi / 4.0
        quaternion = (0.0, 0.0, math.sin(half), math.cos(half))
        expected = quat_rotate_vec(quaternion, (1.0, 0.0, 0.0))
        actual = transform_point(quat_matrix(quaternion), (1.0, 0.0, 0.0))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=12)

    def test_true_mesh_reference_requires_source_visual_link_sync(self) -> None:
        frame_ids = list(MODULE.V3_REQUIRED_FRAME_IDS)
        contract = strict_scene_contract(frame_ids=MODULE.V3_REQUIRED_FRAME_IDS)
        manifest = {
            "mesh_reference_pass": True,
            "capture_ok": True,
            "media_ok": True,
            "asset_ok": True,
            "package_ok": True,
            "data_binding_ok": True,
            "dynamic_sequence_ok": True,
            "source_was_ppm_only": False,
            "frame_ids": frame_ids,
            "expected_frame_ids": frame_ids,
            "scene_contract_v3": contract,
            "render_summary": {
                "visual_kind": "mesh",
                "mesh_detected": True,
                "image_count": len(frame_ids),
                "silhouette_image_count": len(frame_ids),
                "ground_mask_image_count": len(frame_ids),
                "expected_image_count": len(frame_ids),
                "frame_ids": frame_ids,
                "expected_frame_ids": frame_ids,
                "mp4_ok": True,
                "source_was_ppm_only": False,
            },
        }
        self.assertTrue(validate_mesh_reference_manifest(manifest)["ok"])
        manifest["dynamic_sequence_ok"] = False
        self.assertEqual(
            validate_mesh_reference_manifest(manifest)["blocker"],
            "mimickit_dynamic_sequence_failed",
        )
        manifest["dynamic_sequence_ok"] = True
        manifest["capture_ok"] = False
        self.assertEqual(
            validate_mesh_reference_manifest(manifest)["blocker"],
            "mimickit_capture_gate_failed",
        )
        manifest["capture_ok"] = True
        del contract["camera_samples"][0]["visual_link_sync_ok"]
        self.assertEqual(
            validate_mesh_reference_manifest(manifest)["blocker"],
            "mimickit_visual_link_sync_failed",
        )

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
            binding = validate_mesh_binding(asset, joint_order, structure, mesh_binding_contract=binding_contract(["pelvis", "sword", "shield"]))
            self.assertTrue(binding["mesh_binding_pass"], binding)
            self.assertFalse(binding["xml_geom_fallback_used"])

            skinned_asset = root / "skinned.glb"
            valid_skinned_glb(skinned_asset)
            skinned_structure = inspect_glb(skinned_asset)
            skinned_binding = validate_mesh_binding(skinned_asset, joint_order, skinned_structure, mesh_binding_contract=binding_contract(["pelvis", "sword", "shield", "avatar_mesh"]))
            self.assertTrue(skinned_binding["mesh_binding_pass"], skinned_binding)
            self.assertEqual(skinned_binding["binding_mode"], "skinned")

    def test_wrong_binding_contract_and_missing_media_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "valid.glb"
            valid_rigid_glb(asset)
            joint_order = {"body_order": ["pelvis", "sword", "shield"], "dof_size": 1, "joints": []}
            wrong_binding = validate_mesh_binding(
                asset,
                joint_order,
                inspect_glb(asset),
                mesh_binding_contract=binding_contract(["pelvis", "sword"]),
            )
            self.assertFalse(wrong_binding["mesh_binding_pass"])
            self.assertIn("shield", wrong_binding["missing_required_bodies"])

            frame_ids = list(MODULE.V3_REQUIRED_FRAME_IDS)
            manifest = {
                "mesh_reference_pass": True,
                "capture_ok": True,
                "media_ok": True,
                "asset_ok": True,
                "package_ok": True,
                "data_binding_ok": True,
                "dynamic_sequence_ok": True,
                "source_was_ppm_only": False,
                "frame_ids": frame_ids,
                "expected_frame_ids": frame_ids,
                "scene_contract_v3": strict_scene_contract(frame_ids=MODULE.V3_REQUIRED_FRAME_IDS),
                "render_summary": {
                    "visual_kind": "mesh",
                    "mesh_detected": True,
                    "image_count": len(frame_ids),
                    "silhouette_image_count": len(frame_ids),
                    "ground_mask_image_count": len(frame_ids),
                    "expected_image_count": len(frame_ids),
                    "frame_ids": frame_ids,
                    "expected_frame_ids": frame_ids,
                    "mp4_ok": False,
                    "source_was_ppm_only": False,
                },
            }
            missing_media = validate_mesh_reference_manifest(manifest)
            self.assertFalse(missing_media["ok"])
            self.assertEqual(missing_media["blocker"], "mimickit_mp4_invalid")
            manifest["render_summary"]["mp4_ok"] = True
            manifest["render_summary"]["source_was_ppm_only"] = True
            ppm_only = validate_mesh_reference_manifest(manifest)
            self.assertFalse(ppm_only["ok"])
            self.assertEqual(ppm_only["blocker"], "ppm_only_output_rejected")

    def test_scene_contract_hash_and_camera_coverage_are_required(self) -> None:
        contract = strict_scene_contract()
        report = compare_scene_contract(contract, width=64, height=64, stride=5, fps=12, frame_ids=[0, 5])
        self.assertTrue(report["scene_contract_compare_pass"], report)
        applied = report["applied_scene_contract"]
        self.assertEqual(applied["renderer"], "evih_stdlib_glb_triangle_rasterizer_v3")
        self.assertEqual(applied["capture"]["settle_updates_per_attempt"], [0])
        self.assertFalse(applied["capture"]["requires_visual_link_sync"])
        self.assertEqual(applied["camera_samples"][0]["renderer_version"], "evih_stdlib_glb_triangle_rasterizer_v3")
        self.assertNotIn("visual_link_sync_ok", applied["camera_samples"][0])
        contract["camera_samples"].pop()
        report = compare_scene_contract(contract, width=64, height=64, stride=5, fps=12, frame_ids=[0, 5])
        self.assertFalse(report["scene_contract_compare_pass"])
        contract = strict_scene_contract()
        contract["camera_samples"][0]["capture_settle_updates"] = 0
        contract["scene_contract_sha256"] = scene_contract_sha256(contract)
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
            frame_ids = list(MODULE.V3_REQUIRED_FRAME_IDS)
            rows = [
                {
                    "frame": frame,
                    "root_pos_m": [frame * 0.003, 0.0, 0.0],
                    "root_rot_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "dof_pos": [0.0],
                }
                for frame in frame_ids
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
            body_rows = [
                {
                    "frame": row["frame"],
                    "source": "canonical_source_rig_fk_v3",
                    "body_order": joint_order["body_order"],
                    "body_pos_m": [row["root_pos_m"][0], 0.0, 0.0] * 3,
                    "body_rot_xyzw": [0.0, 0.0, 0.0, 1.0] * 3,
                }
                for row in rows
            ]
            (package / "visual_replay" / "body_world_replay.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in body_rows),
                encoding="utf-8",
            )
            for path, payload in (
                (package / "visual_replay" / "pose_dof_meta.json", {"schema_version": 1}),
                (
                    package / "visual_replay" / "body_world_contract.json",
                    {"schema_version": 1, "source": "canonical_source_rig_fk_v3"},
                ),
                (package / "joint_order.json", joint_order),
                (
                    package / "mimickit_source_rig_asset_spec.json",
                    {"schema_version": 1, "flat_bodies": [{"name": name, "parent": ""} for name in joint_order["body_order"]]},
                ),
                (package / "visual_alignment_contract.json", {"schema_version": 1}),
                (package / "mesh_binding_contract.json", binding_contract(joint_order["body_order"])),
            ):
                path.write_text(json.dumps(payload), encoding="utf-8")
            asset = root / "valid.glb"
            valid_rigid_glb(asset)
            contract = strict_scene_contract(frame_ids=MODULE.V3_REQUIRED_FRAME_IDS)
            source_render = root / "source_render"
            binding = validate_mesh_binding(asset, joint_order, inspect_glb(asset), mesh_binding_contract=binding_contract(joint_order["body_order"]))
            source_rows = [dict(row, **body_row) for row, body_row in zip(rows, body_rows)]
            source_rig = {"flat_bodies": [{"name": name, "parent": ""} for name in joint_order["body_order"]]}
            render_true_mesh(source_rows, frame_ids, asset, joint_order, contract, source_render, width=64, height=64, source_rig=source_rig, binding=binding)
            manifest = {
                "mesh_reference_pass": True,
                "capture_ok": True,
                "media_ok": True,
                "asset_ok": True,
                "package_ok": True,
                "dynamic_sequence_ok": True,
                "scene_contract_v3": contract,
                "data_binding_ok": True,
                "data_binding": {
                    "pose_dof_replay_sha256": sha256_file(package / "visual_replay" / "pose_dof_replay.jsonl"),
                    "body_world_replay_sha256": sha256_file(package / "visual_replay" / "body_world_replay.jsonl"),
                },
                "asset_export": {"output_sha256": sha256_file(asset)},
                "render_summary": {
                    "render_dir": str(source_render),
                    "visual_kind": "mesh",
                    "mesh_detected": True,
                    "image_count": len(frame_ids),
                    "silhouette_image_count": len(frame_ids),
                    "ground_mask_image_count": len(frame_ids),
                    "expected_image_count": len(frame_ids),
                    "frame_ids": frame_ids,
                    "expected_frame_ids": frame_ids,
                    "mp4_ok": True,
                    "source_was_ppm_only": False,
                },
            }
            manifest_path = root / "mesh_reference_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = build_true_mesh_replay(
                package_dir=package,
                mesh_asset=asset,
                mesh_reference_manifest_path=manifest_path,
                out_dir=root / "evih",
                thresholds=default_thresholds(),
            )
            self.assertFalse(report["evih_mesh_replay_pass"])
            visual_review = root / "evih" / "visual_review.json"
            review_payload = json.loads(visual_review.read_text(encoding="utf-8"))
            review_payload["visual_review_pass"] = True
            review_payload["reviewer"] = "test-reviewer"
            review_payload["reviewed_at_utc"] = "2026-06-12T00:00:00Z"
            review_payload["checks"] = {name: True for name in MODULE.REQUIRED_VISUAL_REVIEW_CHECKS}
            visual_review.write_text(json.dumps(review_payload), encoding="utf-8")
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
